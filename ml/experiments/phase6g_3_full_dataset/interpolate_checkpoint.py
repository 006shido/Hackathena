"""Weight-space interpolation; never blend two RGB faces."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fraction',type=float,default=0.5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.fraction<1:raise ValueError('Fraction must be between zero and one.')
    if args.output.exists():raise FileExistsError(args.output)
    directory=Path(__file__).parent
    original=torch.load(directory/'run/best_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
    candidate=torch.load(directory/'crossview_pilot/latest_model.pt',map_location='cpu',weights_only=False)['model_state_dict']
    assert original.keys()==candidate.keys()
    blended={}
    for key, value in original.items():
        other=candidate[key]
        if key.startswith('arcface.'):
            assert torch.equal(value,other),f'Frozen identity extractor changed: {key}'
            blended[key]=value
        elif value.is_floating_point():
            blended[key]=torch.lerp(value,other,args.fraction)
        else:
            blended[key]=other
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state_dict=blended,source_checkpoints=['run/best_model.pt','crossview_pilot/latest_model.pt'],
        interpolation_fraction=args.fraction,experimental=True),args.output)
    print(f'Saved experimental weight interpolation {args.fraction}: {args.output}',flush=True)


if __name__=='__main__':main()
