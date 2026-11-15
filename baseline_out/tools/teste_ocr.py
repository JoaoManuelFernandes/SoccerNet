import cv2, numpy as np, torch, time
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox
t=time.time()
det=TextDetInferencer('dbnet_resnet18_fpnc_1200e_icdar2015', device='cuda')
rec=TextRecInferencer('SAR', device='cuda')
print("modelos carregados em %.0fs"%(time.time()-t), flush=True)
img=cv2.imread("/home/joao/soccernet/data/SoccerNetGS/test/SNGS-116/img1/000001.jpg")
crops=[img[565:705,478:529], img[548:688,935:994], img[541:678,981:1050]]   # anotação: 93, 11, 34
d=det(crops, return_datasamples=True, batch_size=8, progress_bar=False)['predictions']
for i,(c,ds) in enumerate(zip(crops,d)):
    polys=ds.pred_instances['polygons']; ins=[]
    for p in polys:
        q=bbox2poly(poly2bbox(p)).tolist(); ci=crop_img(c,q)
        if ci.shape[0] and ci.shape[1]: ins.append(ci)
    if ins:
        r=rec(ins, return_datasamples=True, batch_size=8, progress_bar=False)['predictions']
        print(i, "polígonos:",len(polys), [(x.pred_text.item, round(float(np.mean(x.pred_text.score)),3)) for x in r], flush=True)
    else: print(i, "polígonos:",len(polys), "sem texto", flush=True)
print("VRAM pico %.0f MB"%(torch.cuda.max_memory_allocated()/2**20))
