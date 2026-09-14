"""Fixed model fits with per-fit resume, immutable attempts and full provenance."""
from common import *
import functools, time, concurrent.futures
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.data import load_uci_har, load_wisdm_arff
from src.splits import make_rotating_splits
from src.models import build_model
from src.experiment import aligned_predict_proba, fit_with_thread_limit, split_digest
CONFIG=json.loads((ROOT/"configs"/"expanded.json").read_text(encoding="utf-8"))
CONFIG_HASH=sha256(ROOT/"configs"/"expanded.json")
@functools.lru_cache(maxsize=2)
def features(dataset):
    if dataset=="uci_har": return load_uci_har(ROOT/"work"/"data"/"uci_har_240"/"UCI HAR Dataset")
    return load_wisdm_arff(ROOT/"work"/"data"/"wisdm_507"/"wisdm-dataset")
@functools.lru_cache(maxsize=2)
def raw_data(dataset):
    from deep_har import load_uci_raw, load_wisdm_raw
    cache=ROOT/"work"/"cache"/(dataset+"_raw.joblib")
    import joblib
    if cache.exists(): return joblib.load(cache)
    if dataset=="uci_har": data=load_uci_raw(ROOT/"work"/"data"/"uci_har_240"/"UCI HAR Dataset")
    else: data=load_wisdm_raw(ROOT/"work"/"data"/"wisdm_507"/"wisdm-dataset",allowed_subjects=np.unique(features(dataset).subjects))
    cache.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(data,cache,compress=3)
    atomic_json(ROOT/"01_environment"/(dataset+"_raw_provenance.json"),data["metadata"])
    return data
def fit_directory(dataset,model,protocol,seed,fold):
    section="03_deep_har" if model=="cnn1d" else "06_robustness"
    return ROOT/section/"fits"/f"{dataset}__{protocol}__s{seed}__f{fold}__{model}"
def folds(dataset,model,protocol,seed):
    b=features(dataset); encoder=LabelEncoder().fit(b.y)
    base=make_rotating_splits(encoder.transform(b.y),b.subjects,5,seed,protocol)
    if model!="cnn1d":
        return [(s,{k:getattr(s,k) for k in ["train","calibration","test"]}) for s in base]
    raw=raw_data(dataset)
    if protocol=="sample_mixed":
        encoded=LabelEncoder().fit_transform(raw["y"])
        base=make_rotating_splits(encoded,raw["subjects"],5,seed,protocol)
        return [(s,{k:getattr(s,k) for k in ["train","calibration","test"]}) for s in base]
    mapped=[]
    for s in base:
        roles={k:np.flatnonzero(np.isin(raw["subjects"],np.unique(b.subjects[getattr(s,k)]))) for k in ["train","calibration","test"]}
        mapped.append((s,roles))
    return mapped
def internal_validation(roles,subjects,y,protocol,seed):
    roles={k:np.asarray(v,dtype=int) for k,v in roles.items()}
    train=roles["train"]
    if protocol=="subject_disjoint":
        users=np.unique(subjects[train])
        rng=np.random.default_rng(seed)
        n=max(1,int(np.ceil(.2*len(users))))
        if n>=len(users): raise ValueError("Insufficient training participants for validation")
        selected=rng.choice(users,size=n,replace=False)
        val=train[np.isin(subjects[train],selected)]
        fit=train[~np.isin(subjects[train],selected)]
    else:
        fit,val=train_test_split(train,test_size=.2,random_state=seed,stratify=y[train])
    roles.update(train=np.sort(fit),validation=np.sort(val))
    return roles
def valid_completed(path):
    marker=path/"completed.json"
    if not marker.exists(): return False
    value=json.loads(marker.read_text(encoding="utf-8"))
    if value["config_sha256"]!=CONFIG_HASH: raise RuntimeError("Frozen configuration changed for "+str(path))
    for item in value["artifacts"]:
        p=ROOT/item["path"]
        if not p.is_file() or sha256(p)!=item["sha256"]: raise RuntimeError("Completed fit artifact missing or changed: "+str(p))
    return True
def fit_one(spec):
    dataset,model,protocol,seed,fold=spec
    out=fit_directory(*spec)
    if valid_completed(out): return str(out)
    from reliability import assert_roles,validate_probabilities
    started=now(); clock=time.perf_counter()
    attempt=out/("attempt_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
    attempt.mkdir(parents=True,exist_ok=False)
    identity=dict(zip(["dataset","model","protocol","seed","fold"],spec))
    meta={**identity,"config_sha256":CONFIG_HASH,"started_utc":started,"ended_utc":None,"environment":environment(),"status":"running","exception":None,"calibration_method":"none at base-model fitting","hyperparameters":CONFIG["deep"] if model=="cnn1d" else None}
    atomic_json(attempt/"metadata.json",meta)
    log("FIT_START",**identity,path=str(attempt))
    try:
        b=features(dataset)
        data=raw_data(dataset) if model=="cnn1d" else {"X":b.X,"y":b.y,"subjects":b.subjects,"sample_ids":b.sample_ids,"metadata":b.metadata}
        encoder=LabelEncoder().fit(b.y)
        y=encoder.transform(data["y"]); subjects=np.asarray(data["subjects"]); ids=np.asarray(data["sample_ids"]); X=data["X"]
        _,roles=folds(dataset,model,protocol,seed)[fold]
        if model=="cnn1d": roles=internal_validation(roles,subjects,y,protocol,seed+fold)
        assert_roles(subjects,roles,subject_disjoint=protocol=="subject_disjoint")
        meta["classes"]=encoder.classes_.tolist()
        meta["participant_split"]={k:np.unique(subjects[idx]).tolist() for k,idx in roles.items()}
        meta["n_rows"]={k:len(idx) for k,idx in roles.items()}
        meta["representation"]="raw_sensor_windows" if model=="cnn1d" else "V1_official_features"
        meta["representation_metadata"]=data["metadata"]
        np.savez_compressed(attempt/"roles.npz",**roles)
        if model=="cnn1d":
            from deep_har import fit_cnn
            result=fit_cnn(X,y,roles,seed+fold,attempt/"model.pt",CONFIG["deep"])
            pcal=result["calibration_probabilities"]; ptest=result["test_probabilities"]
            logitscal=result["calibration_logits"]; logitstest=result["test_logits"]
            meta["training"]=result["metadata"]
            atomic_json(attempt/"training_history.json",result["history"])
        else:
            import joblib
            estimator=build_model(model,seed+fold,n_jobs=CONFIG["classical_n_jobs"],pilot=False)
            meta["hyperparameters"]={k:str(v) if not isinstance(v,(str,int,float,bool,type(None))) else v for k,v in estimator.get_params(deep=False).items()}
            fit_with_thread_limit(estimator,X[roles["train"]],y[roles["train"]],limit=1)
            pcal=aligned_predict_proba(estimator,X[roles["calibration"]],len(encoder.classes_))
            ptest=aligned_predict_proba(estimator,X[roles["test"]],len(encoder.classes_))
            logitscal=np.log(np.clip(pcal,1e-12,1)); logitstest=np.log(np.clip(ptest,1e-12,1))
            meta["score_kind"]="clipped log probabilities; not native classifier logits"
            joblib.dump(estimator,attempt/"model.joblib",compress=3)
        for p in (pcal,ptest): validate_probabilities(p)
        for partition,p,scores in [("calibration",pcal,logitscal),("test",ptest,logitstest)]:
            idx=roles[partition]
            if not np.isfinite(scores).all(): raise ValueError("Nonfinite scores")
            np.savez_compressed(attempt/(partition+"_raw.npz"),probabilities=p,logits=scores,y=y[idx],subjects=subjects[idx],sample_ids=ids[idx],row_indices=idx,classes=encoder.classes_)
        meta.update(status="completed",ended_utc=now(),elapsed_seconds=time.perf_counter()-clock,output_paths=[str(p.relative_to(ROOT)) for p in attempt.iterdir()])
        atomic_json(attempt/"metadata.json",meta)
        artifacts=[{"path":str(p.relative_to(ROOT)),"sha256":sha256(p)} for p in attempt.iterdir() if p.is_file()]
        atomic_json(out/"completed.json",{"config_sha256":CONFIG_HASH,"attempt":str(attempt.relative_to(ROOT)),"artifacts":artifacts,"finished_utc":now()})
        log("FIT_COMPLETED",**identity,elapsed_seconds=meta["elapsed_seconds"])
        return str(out)
    except Exception:
        meta.update(status="failed",ended_utc=now(),exception=traceback.format_exc())
        atomic_json(attempt/"metadata.json",meta)
        log("FIT_FAILED",**identity,traceback=meta["exception"])
        raise
def all_specs(models,seeds):
    return [(dataset,model,protocol,int(seed),fold) for seed in seeds for dataset in CONFIG["datasets"] for protocol in CONFIG["protocols"] for fold in range(5) for model in models]
def run_fits(models,seeds):
    specs=all_specs(models,seeds)
    errors=[]
    if models==["cnn1d"]:
        for spec in specs:
            try: fit_one(spec)
            except Exception as exc: errors.append({"spec":spec,"error":str(exc)})
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["classical_workers"]) as pool:
            future_specs={pool.submit(fit_one,s):s for s in specs}
            for future in concurrent.futures.as_completed(future_specs):
                try: future.result()
                except Exception as exc: errors.append({"spec":future_specs[future],"error":str(exc)})
    if errors:
        atomic_json(ROOT/"logs"/("fit_errors_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+".json"),errors)
        raise RuntimeError(f"{len(errors)} model fits failed; all failures retained")
    return {"completed_fits":len(specs),"models":models,"seeds":list(seeds)}
def successful_fits():
    records=[]
    for section in ["03_deep_har","06_robustness"]:
        for marker in sorted((ROOT/section/"fits").glob("*/completed.json")):
            info=json.loads(marker.read_text(encoding="utf-8"))
            records.append(ROOT/info["attempt"])
    return records
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--classical",action="store_true"); parser.add_argument("--seed",type=int,default=20260911)
    args=parser.parse_args()
    run_fits(CONFIG["models"][:-1] if args.classical else ["cnn1d"],[args.seed])
