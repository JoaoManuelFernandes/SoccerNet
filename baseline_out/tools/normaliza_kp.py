"""Repõe os keypoints normalizados (0-1) num estado em que a calibração original os desnormalizou no
próprio sítio (FramebyFrameCalib.update com denormalize=True altera o dict em memória, e o estado
guarda-o assim). Só toca nas imagens em píxeis. Uso: normaliza_kp.py <in.pklz> <out.pklz>

Critério: uma imagem está em píxeis se o MAIOR valor de x ou y entre todos os keypoints for > 3.0.
(A versão anterior usava "algum valor > 1.0", o que apanhava imagens normalizadas com um keypoint
ligeiramente fora do enquadramento, x ou y em ]1.0, 1.5], e dividia-as por 1920x1080 uma segunda vez,
o que corrompia a homografia dessas frames; corrigido a 2026-09-21.)"""
import sys, zipfile, pickle, os
W, H = 1920.0, 1080.0
src, dst = sys.argv[1], sys.argv[2]
zi = zipfile.ZipFile(src); n_fix = n_img = 0
with zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_DEFLATED) as zo:
    for name in zi.namelist():
        data = zi.read(name)
        if name.endswith("_image.pkl"):
            df = pickle.loads(data)
            for idx, kp in df["keypoints"].items():
                if isinstance(kp, dict) and kp:
                    n_img += 1
                    if max(max(v.get("x", 0.0), v.get("y", 0.0)) for v in kp.values()) > 3.0:
                        for v in kp.values(): v["x"] /= W; v["y"] /= H
                        n_fix += 1
            data = pickle.dumps(df)
        zo.writestr(name, data)
os.replace(dst + ".tmp", dst)
print(f"{os.path.basename(src)}: {n_fix}/{n_img} imagens renormalizadas")
