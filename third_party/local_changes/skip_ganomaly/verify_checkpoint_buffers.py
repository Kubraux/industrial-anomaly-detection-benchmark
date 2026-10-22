import torch,json
from pathlib import Path
old=Path('output/skipganomaly_mvtec_bottle_smoke/train/weights')
new=Path('output/bottle_audit_smoke/train/weights')
for net in ['G','D']:
    a=torch.load(old/('net%s_0.pth'%net),map_location='cpu')['state_dict']
    b=torch.load(new/('net%s_1.pth'%net),map_location='cpu')['state_dict']
    buffers=lambda k: any(s in k for s in ['running_mean','running_var','num_batches_tracked'])
    parameters=[k for k in a if not buffers(k)]
    changed=[k for k in parameters if not torch.equal(a[k],b[k])]
    bd=[k for k in a if buffers(k) and not torch.equal(a[k],b[k])]
    print(net,'parameter tensors',len(parameters),'changed',len(changed),
          'max_parameter_difference',max(float((a[k]-b[k]).abs().max()) for k in parameters),
          'changed_buffers',len(bd))
    key=next(k for k in a if 'num_batches_tracked' in k)
    print('first BN count',int(a[key]),'->',int(b[key]),flush=True)
