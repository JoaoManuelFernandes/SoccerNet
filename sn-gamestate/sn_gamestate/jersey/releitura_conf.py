"""Relê as janelas do ensaio com o Qwen a devolver a CONFIANÇA real (probabilidade dos tokens da resposta).
Mesmo modelo, mesmo carregamento 4-bit, mesmo prompt e mesma descodificação do módulo; uma janela por
chamada (solo), até 5 crops mais nítidos, como o módulo faz. As janelas reconstroem-se dos crops marcados
'qwen*' (os que foram enviados), por ordem de frame, de 6 em 6 — é assim que o módulo as forma.
Saída: <ensaio>/releitura/<vídeo>.jsonl  (retoma se interrompido). Uso: releitura_conf.py <ensaio> <vídeos...>"""
import sys, os, re, json, glob, math, time
sys.path.insert(0, "/home/joao/soccernet/sn-gamestate")
import cv2, numpy as np, torch
from sn_gamestate.jersey.tracklets_jnr_v2 import _load_qwen_model, _clean_qwen_output
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"; EXP=sys.argv[1]; VIDS=sys.argv[2:]
model, proc = _load_qwen_model(MODEL)
from qwen_vl_utils import process_vision_info
from PIL import Image
def sharp(im): return float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
PROMPT=("I'm showing you 1 groups of jersey images. Each group shows the BACK of the same sports player from different video frames.\n"
 "Task: for each group, identify the jersey number printed on the back.\n\nOutput format — exactly 1 lines, no other text:\n"
 "group0: <number or none>\n\nRules:\n- Output ONLY 1-2 digits (e.g. 7, 28) or the word 'none'.\n"
 "- Always output the COMPLETE number — never drop a digit. If you see '1' followed by '4', output '14', not '4'.\n"
 "- Jersey numbers are 1–99. A single digit like '4' is valid, but first check if there is a leading digit partially visible at the edge (e.g. '14', '24', '34').\n"
 "- Only output a number if you are CONFIDENT you can read every digit clearly. If the images are too small, blurry, at a bad angle, or you would be guessing, output 'none' instead — a wrong guess is worse than admitting you can't read it.\n"
 "- NO markdown, NO quotes, NO explanations.")
SYS=("You are a jersey number extractor. Your job is to read the COMPLETE number on the back of a sports jersey. "
     "Always output all visible digits together. Never truncate a number.")
def le(crops):
    content=[{"type":"text","text":f"Group 0 ({len(crops)} frames):"}]
    for c in crops: content.append({"type":"image","image":Image.fromarray(cv2.cvtColor(c,cv2.COLOR_BGR2RGB))})
    content.append({"type":"text","text":PROMPT})
    msgs=[{"role":"system","content":SYS},{"role":"user","content":content}]
    text=proc.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ii,vi=process_vision_info(msgs)
    inp=proc(text=[text],images=ii,videos=vi,padding=True,return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out=model.generate(**inp,max_new_tokens=20,do_sample=False,temperature=None,top_p=None,repetition_penalty=1.1,
                           pad_token_id=proc.tokenizer.eos_token_id,eos_token_id=proc.tokenizer.eos_token_id,
                           output_scores=True,return_dict_in_generate=True)
    gen=out.sequences[0, inp["input_ids"].shape[1]:]
    toks=[proc.tokenizer.decode([t]) for t in gen]
    probs=[float(torch.softmax(s[0].float(),-1)[t]) for s,t in zip(out.scores, gen)]
    reply=proc.tokenizer.decode(gen,skip_special_tokens=True).strip()
    m=re.match(r'group\s*\d+\s*[:\-]\s*(.+)', reply.split("\n")[0].strip(), re.I)
    ans=_clean_qwen_output(m.group(1).strip()) if m else None
    # confiança = produto das probabilidades dos tokens da resposta (depois dos ':' até ao fim da linha)
    i=next((k for k,s in enumerate(toks) if ":" in s), None); ps=[]
    if i is not None:
        for s,p in zip(toks[i+1:], probs[i+1:]):
            if "\n" in s or s.strip()=="" and ps: break
            if s.strip(): ps.append(p)
    conf=math.prod(ps) if ps else 0.0
    return ans, conf, (min(ps) if ps else 0.0), reply
for v in VIDS:
    base=f"{EXP}/debug/{v}"; os.makedirs(f"{EXP}/releitura",exist_ok=True); outp=f"{EXP}/releitura/{v}.jsonl"
    feitas=set()
    if os.path.exists(outp):
        for l in open(outp): r=json.loads(l); feitas.add((r["tid"],r["janela"]))
    t0=time.time(); n=0
    with open(outp,"a") as fo:
        for td in sorted(glob.glob(f"{base}/tracklet_*")):
            tid=int(float(td.split("tracklet_")[1]))
            fs=sorted(glob.glob(f"{td}/frame_*_qwen*.jpg"), key=lambda p:int(re.search(r"frame_(\d+)_",p).group(1)))
            for w in range(0,len(fs),6):
                if (tid,w//6) in feitas: continue
                grupo=fs[w:w+6]; ims=[cv2.imread(p) for p in grupo]; ims=[x for x in ims if x is not None]
                if not ims: continue
                top=sorted(ims,key=sharp,reverse=True)[:5]
                tags=[re.search(r"_qwen(.+)\.jpg",p).group(1) for p in grupo]
                ans,conf,pmin,reply=le(top)
                fo.write(json.dumps(dict(video=v,tid=tid,janela=w//6,n=len(ims),ans=ans,conf=round(conf,4),pmin=round(pmin,4),
                                         tag_log=max(set(tags),key=tags.count),reply=reply))+"\n"); n+=1
    print(f"{v}: {n} janelas relidas em {time.time()-t0:.0f}s", flush=True)
print("RELEITURA CONCLUÍDA", flush=True)
