import sys, types, importlib, copy, time
from common import *
import torch
torch.set_num_threads(1)

def official():
    base = HERE / 'vendor/grin/lib'
    for name, path in [('lib',base),('lib.nn',base/'nn'),('lib.nn.layers',base/'nn/layers'),('lib.nn.models',base/'nn/models'),('lib.nn.utils',base/'nn/utils'),('lib.utils',base/'utils')]:
        module=types.ModuleType(name)
        module.__path__=[str(path)]
        sys.modules[name]=module
    sys.modules['lib'].epsilon=1e-8
    layer=importlib.import_module('lib.nn.layers.gril')
    sys.modules['lib.nn.layers'].BiGRIL=layer.BiGRIL
    return importlib.import_module('lib.nn.models.grin').GRINet, '4a28afbb092600b6e6abeeabaaf67e87dbd1ed6e'

def cases(ds, part, count=None):
    ans = []
    bs = blocks(ds, part)
    bs = subset_days(bs, count) if count else bs
    for day, ix in bs:
        for v, station in enumerate(ds['stations']):
            truth = ds['Z'][ix]
            good = np.isfinite(truth[24:, v])
            if good.sum() < 12 * truth.shape[2]:
                continue
            mask = np.isfinite(truth)
            target = np.zeros_like(mask)
            target[24:, v] = good
            mask[24:, v] = False
            x = np.where(mask, truth, 0)
            ans.append((day, str(station), v, x, mask, target, np.nan_to_num(truth)))
    return ans

def arrays(batch):
    return [torch.as_tensor(np.stack([c[i] for c in batch]), dtype=torch.bool if i in [4, 5] else torch.float32) for i in [3, 4, 5, 6]]

def evaluate(model, cs):
    model.eval()
    rows = []
    with torch.no_grad():
        for j in range(0, len(cs), 16):
            batch = cs[j:j + 16]
            x, mask, target, y = arrays(batch)
            pred = model(x, mask=mask)
            for c, pr, tr, g in zip(batch, pred, y, target):
                rows.append({'day': c[0], 'station': c[1], 'mse': float(((pr - tr)[g] ** 2).mean())})
    return rows

def main():
    GRIN, commit = official()
    start = time.time()
    for region in regions():
        ds = load(region)
        m = models(region)
        n, p = ds['Z'].shape[1:]
        adj = np.zeros((n, n), np.float32)
        for u, v in m['edges']:
            adj[u, v] = adj[v, u] = 1
        train = cases(ds, 0, 80)
        val = cases(ds, 1, 12)
        test = cases(ds, 2)
        records = []
        choices = []
        identity = {'source_sha256': sha(__file__), 'dataset_sha256': sha(OUT / region / 'dataset.npz'), 'vendor_commit': commit, 'seeds': [0, 1, 2], 'widths': [32, 64], 'epochs': 30, 'early_stopping': 7, 'train_day_limit': 80, 'validation': 'first half only', 'timing': 'Additional benchmark fixed before neural training, after primary comparison was computed'}
        lock(region + '/neural_lock.json', identity)
        for seed in [0, 1, 2]:
            options = []
            for width in [32, 64]:
                torch.manual_seed(1700 + seed)
                rng = np.random.default_rng(1700 + seed)
                model = GRIN(adj, p, width, width, 0.0, d_emb=8)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                bestloss = np.inf
                best = None
                bestepoch = 0
                first = 0
                finished = False
                training = []
                checkpoint = OUT / region / f'grin_seed{seed}_width{width}_training.pt'
                if checkpoint.exists():
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    if state['identity'] != identity:
                        raise RuntimeError('Checkpoint identity mismatch')
                    model.load_state_dict(state['model'])
                    optimizer.load_state_dict(state['optimizer'])
                    bestloss = state['bestloss']
                    best = state['best']
                    bestepoch = state['bestepoch']
                    first = state['epoch']
                    finished = state['finished']
                    training = state['training']
                    rng.bit_generator.state = state['numpy_rng']
                    torch.set_rng_state(state['torch_rng'])
                for epoch in [] if finished else range(first, 30):
                    model.train()
                    order = rng.permutation(len(train))
                    for j in range(0, len(order), 16):
                        batch = [train[i] for i in order[j:j + 16]]
                        x, mask, target, y = arrays(batch)
                        optimizer.zero_grad()
                        pred, aux = model(x, mask=mask)
                        loss = ((pred - y)[target] ** 2).mean() + sum((((a - y)[target] ** 2).mean() for a in aux)) / len(aux)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                        optimizer.step()
                    vmse = float(np.mean([r['mse'] for r in evaluate(model, val)]))
                    training.append({'epoch': epoch + 1, 'validation_mse': vmse})
                    if vmse < bestloss:
                        bestloss = vmse
                        best = copy.deepcopy(model.state_dict())
                        bestepoch = epoch + 1
                    finished = epoch + 1 - bestepoch >= 7 or epoch + 1 == 30
                    temp = checkpoint.with_suffix('.tmp')
                    torch.save({'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'bestloss': bestloss, 'best': best, 'bestepoch': bestepoch, 'epoch': epoch + 1, 'finished': finished, 'training': training, 'torch_rng': torch.get_rng_state(), 'numpy_rng': rng.bit_generator.state}, temp)
                    temp.replace(checkpoint)
                    save('neural_progress.json', {'status': 'running', 'region': region, 'seed': seed, 'width': width, 'epoch': epoch + 1, 'elapsed_seconds': time.time() - start, 'pid': os.getpid()})
                    print(region, 'GRIN', seed, width, 'epoch', epoch + 1, 'validation', round(vmse, 5), flush=True)
                    if finished:
                        break
                model.load_state_dict(best)
                options.append((bestloss, model, width, bestepoch))
                frame_save(region + f'/grin_seed{seed}_width{width}_training.csv', pd.DataFrame(training))
            loss, model, width, epoch = min(options, key=lambda x: x[0])
            choice = {'seed': seed, 'width': width, 'epoch': epoch, 'validation_mse': loss, 'training_cases': len(train), 'validation_cases': len(val)}
            choices.append(choice)
            records.extend([{**r, 'seed': seed, 'method': 'grin_retrospective'} for r in evaluate(model, test)])
            frame_save(region + '/grin_cases.csv', pd.DataFrame(records))
            save(region + '/grin_choices.json', choices)
        f = pd.DataFrame(records)
        reference = pd.read_csv(OUT / region / 'outage_cases.csv', dtype={'station': str})
        reference = reference[reference.method == 'sheaf_retrospective']
        summary = {str(seed): paired(pd.concat([f[f.seed == seed], reference]), 'sheaf_retrospective', 'grin_retrospective') for seed in [0, 1, 2]}
        save(region + '/grin_summary.json', {'status': 'completed', 'comparisons': summary, 'choices': choices, 'scope': 'Official pinned bidirectional architecture; retrospective only; all seeds retained; project-specific training not original published scores'})
    save('neural_progress.json', {'status': 'completed', 'elapsed_seconds': time.time() - start})
