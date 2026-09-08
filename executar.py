"""Executa os cadernos e registra separadamente erro de execução e etapa pendente.

Exemplos:
    python executar.py --kernel mm845
    python executar.py --kernel mm845 2 8
"""
from pathlib import Path
import argparse
import os
import sys
import time
import json
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parent

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tutoriais',nargs='*',type=int,help='números de 1 a 8; omitir para executar todos')
    parser.add_argument('--kernel',default='python3',help='nome do kernel Jupyter (exemplo: mm845)')
    args=parser.parse_args()
    if any(i not in range(1,9) for i in args.tutoriais):parser.error('Escolha somente tutoriais de 1 a 8.')
    failed=False
    for i in args.tutoriais or range(1,9):
        path=ROOT/'estudos'/f'tutorial_{i:02}'/'solucoes.ipynb'
        nb=nbformat.read(path,as_version=4);start=time.time()
        print(f'INÍCIO tutorial {i:02}',flush=True)
        try:
            NotebookClient(nb,timeout=900,kernel_name=args.kernel,
                           resources={'metadata':{'path':str(path.parent)}}).execute()
            errors=[o for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]) if o.output_type=='error']
            if errors:raise RuntimeError('Há saídas de erro no caderno.')
            record={'status':'executado_sem_erros','segundos':round(time.time()-start,2),
                    'celulas_codigo':sum(c.cell_type=='code' for c in nb.cells),
                    'celulas_executadas':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
                    'cobertura_experimental':'completa para o caderno de soluções'}
            stage=path.parent/'resultados/estado_umap.json'
            if i==8 and stage.exists():
                state=json.loads(stage.read_text())
                if state.get('status')!='EXECUTADO':
                    record['cobertura_experimental']='parcial: UMAP não executado'
                    record['pendencias']=[state]
            print('OK',i,record,flush=True)
        except Exception as exc:
            failed=True
            record={'status':'falhou','erro':str(exc),'segundos':round(time.time()-start,2)}
            print('ERRO',i,str(exc),flush=True)
        finally:
            nbformat.write(nb,path)
            (path.parent/'validacao.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    report_path=ROOT/'VALIDACAO.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    report['tutoriais']={}
    for i in range(1,9):
        p=ROOT/'estudos'/f'tutorial_{i:02}'/'validacao.json'
        if p.exists():report['tutoriais'][f'tutorial_{i:02}']=json.loads(p.read_text())
    report['celulas_codigo_executadas']=sum(s.get('celulas_executadas',0) for s in report['tutoriais'].values())
    report['pendencias_atuais']=[p for s in report['tutoriais'].values() for p in s.get('pendencias',[])]
    report['ultima_reexecucao_local']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return int(failed)

if __name__=='__main__':sys.exit(main())
