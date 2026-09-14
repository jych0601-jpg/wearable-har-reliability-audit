"""Download exactly the V1 official archives and safely restore inputs."""
from pathlib import Path
import concurrent.futures, hashlib, json, os, time, traceback, urllib.request, zipfile
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work" / "data"
SPECS = [
("uci_har_240", "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip", "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031", "UCI HAR Dataset.zip", "2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0"),
("wisdm_507", "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip", "6e0147f181d8f5275918db65eeb9dc842a0185d37cb6b2f30cbf74ee6e205469", "wisdm-dataset.zip", "1cf32e8d3a0ecc101bd299fba22545273dcb7de1c9565e0b7844abdfbc6b1695")]
def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
def extract(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            target = (dest / item.filename).resolve()
            if not target.is_relative_to(dest.resolve()):
                raise ValueError("Unsafe archive member: " + item.filename)
        bad = z.testzip()
        if bad: raise ValueError("Corrupt ZIP member: " + bad)
        z.extractall(dest)
def one(spec):
    name,url,sha,inner,inner_sha = spec
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / (name + ".zip")
    log = ROOT / "logs" / ("download_" + name + ".log")
    with log.open("a", encoding="utf-8", buffering=1) as f:
        f.write(json.dumps({"time":time.time(),"url":url,"expected_sha256":sha})+"\n")
        if not archive.exists():
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent":"HAR-Reproducibility-Audit/1.0"})
                    with urllib.request.urlopen(req, timeout=120) as response, archive.with_suffix(".partial").open("wb") as target:
                        while chunk := response.read(1024*1024): target.write(chunk)
                    if digest(archive.with_suffix(".partial")) != sha:
                        raise ValueError("Downloaded source checksum mismatch")
                    os.replace(archive.with_suffix(".partial"), archive)
                    break
                except Exception:
                    f.write(traceback.format_exc()+"\n")
                    if attempt == 2: raise
                    time.sleep(2*(attempt+1))
        if digest(archive) != sha: raise ValueError("Existing source checksum mismatch")
        dest=DATA/name
        marker=dest/"restore_completed.json"
        if not marker.exists():
            extract(archive,dest)
            nested=dest/inner
            if digest(nested) != inner_sha: raise ValueError("Nested archive checksum mismatch")
            extract(nested,dest)
            marker.write_text(json.dumps({"source_sha256":sha,"inner_sha256":inner_sha,"finished":time.time()}),encoding="utf-8")
        f.write(json.dumps({"time":time.time(),"status":"completed","path":str(dest)})+"\n")
        return str(dest)
if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(one,SPECS): print(result,flush=True)
