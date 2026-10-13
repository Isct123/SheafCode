                                                                                      
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).parent))
from operational_guard import assess

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);args=parser.parse_args()
    record=json.loads(args.input.read_text(encoding='utf-8-sig'))
    if record.get('reading_kind')!='measured':raise ValueError('Explicit reading_kind="measured" required; imputed values are not queried measurements')
    result=assess(record['stations'],record['readings'],record['threshold'],record['noise'],record['faults'])
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError) as error:
        print(f'INVALID INPUT: {error}',file=sys.stderr);sys.exit(1)
