from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import wfdb
from contextlib import asynccontextmanager

DATA_ROOT = Path(r"D:\Documentos\Cursos\Diplomatura IA\Modulo3\Redes Neuronales para el Análisis de Series Temporales\Proyecto\ECG\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords")

records_index = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global records_index
    records_index = []

    for hea in DATA_ROOT.rglob("*.hea"):
        base = hea.with_suffix("")
        try:
            h = wfdb.rdheader(str(base))
            sig_names = h.sig_name if getattr(h, "sig_name", None) else [f"ch{i+1}" for i in range(h.nsig)]
            try:
                folder_rel = str(hea.parent.relative_to(DATA_ROOT))
            except Exception:
                folder_rel = str(hea.parent)
            records_index.append({
                "id": hea.stem,
                "base": str(base),
                "carpeta": folder_rel,
                "fs": float(getattr(h, "fs", 0)),
                "sig_len": int(getattr(h, "sig_len", 0)),
                "n_sig": int(getattr(h, "nsig", 0)),
                "sig_names": ",".join(sig_names),
            })
        except Exception:
            continue

    print(f"Índice cargado: {len(records_index)} registros")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/record/{record_id}/{ext}")
def get_record_file(record_id: str, ext: str):
    if ext not in ["hea", "mat"]:
        raise HTTPException(status_code=400, detail="Extensión no válida")

    pattern = list(DATA_ROOT.rglob(f"{record_id}.{ext}"))
    if not pattern:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    return FileResponse(pattern[0], media_type='application/octet-stream')


@app.get("/records")
def get_records_index():
    return JSONResponse(content=records_index)
