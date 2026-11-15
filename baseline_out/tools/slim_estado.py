"""Reduz um estado .pklz da baseline: retira embeddings, embeddings_with_colors e body_masks
(só servem ao tracking/ReID, já feitos). Mantém tudo o que a avaliação, o GS-HOTA e a tabela usam.
Uso: slim_estado.py <origem.pklz> <destino.pklz>"""
import sys, zipfile, pickle, io, os
src, dst = sys.argv[1], sys.argv[2]
size0 = os.path.getsize(src)
DROP = {"embeddings", "embeddings_with_colors", "body_masks", "visibility_scores"}
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_DEFLATED) as zout:
    for n in zin.namelist():
        data = zin.read(n)
        if n.endswith(".pkl") and not n.endswith("_image.pkl"):
            df = pickle.load(io.BytesIO(data))
            df = df.drop(columns=[c for c in DROP if c in df.columns])
            data = pickle.dumps(df)
        elif n == "summary.json":   # o carregador do tracklab exige as colunas listadas aqui
            import json
            sm = json.loads(data)
            cols = sm.get("columns")
            if isinstance(cols, list): sm["columns"] = [c for c in cols if c not in DROP]
            elif isinstance(cols, dict): sm["columns"] = {k: [c for c in v if c not in DROP] for k, v in cols.items()}
            data = json.dumps(sm).encode()
        zout.writestr(n, data)
os.replace(dst + ".tmp", dst)
print(f"{os.path.basename(src)}: {size0/2**20:.0f} MB -> {os.path.getsize(dst)/2**20:.1f} MB")
