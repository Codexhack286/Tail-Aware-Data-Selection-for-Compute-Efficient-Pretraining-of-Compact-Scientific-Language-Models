"""Run a bounded output-masked drafting fine-tuning pilot."""
import argparse, json, random, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from model import GPT

ROOT = Path(__file__).resolve().parent.parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT/'data'/'draft_sft_v1')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'runs'/'random'/'seed42'/'ckpt_best.pt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-5)
    args = p.parse_args()
    if args.steps < 1: p.error('steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, ckpt = GPT.from_checkpoint(args.checkpoint)
    model.to(device)
    tok = Tokenizer.from_file(str(ROOT/'tokenizer'/'tokenizer.json'))
    eot = tok.token_to_id('<|endoftext|>')
    def load(split):
        rows = [json.loads(line) for line in (args.data/f'{split}.jsonl').read_text(encoding='utf8').splitlines()]
        if not rows: raise ValueError(f'empty {split} set')
        return rows
    train, val = load('train'), load('val')
    def loss(row):
        prefix = tok.encode(row['prompt']).ids
        target = tok.encode(row['target']).ids + [eot]
        ids = prefix + target
        if len(ids) > model.config.block_size: raise ValueError('example exceeds context')
        x = torch.tensor([ids[:-1]],device=device)
        y = torch.tensor([[(-100 if i < len(prefix) else ids[i]) for i in range(1,len(ids))]],device=device)
        return model(x,y)[1]
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)
    start = time.perf_counter()
    with (args.output/'metrics.jsonl').open('x', encoding='utf8') as log:
        for step in range(1,args.steps+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            l = loss(random.choice(train)); l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
            if step == 1 or step % 25 == 0 or step == args.steps:
                model.eval()
                with torch.no_grad(): v = sum(float(loss(r)) for r in val[:min(10,len(val))])/min(10,len(val))
                record = {'step':step,'train_loss':float(l),'val_loss':v,'elapsed_s':time.perf_counter()-start}
                log.write(json.dumps(record)+'\n'); log.flush(); print(record,flush=True)
    torch.save({'model':model.state_dict(),'config':ckpt['config'],'step':args.steps,'draft_sft':True},args.output/'ckpt_best.pt')
    (args.output/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf8')

if __name__ == '__main__': main()
