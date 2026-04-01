import yaml
import pathlib
import pandas as pd

# ── Write correct config ──────────────────────
cfg = {
    'project': {'name': 'MedViT-MT', 'seed': 42, 'device': 'cpu'},
    'paths': {
        'dataset_root':       'E:/Brain Tumer/Dataset',
        'mri_images':         'E:/Brain Tumer/Dataset/MRI Images',
        'clinical_csv':       'E:/Brain Tumer/Dataset/UPENN-GBM_clinical_info_v2_1.csv',
        'mri_subfolder':      '',
        'processed_images':   'E:/Brain Tumer/data/processed/images',
        'processed_clinical': 'E:/Brain Tumer/data/processed/clinical',
        'splits_dir':         'E:/Brain Tumer/data/splits',
        'cache_dir':          'E:/Brain Tumer/data/cache',
        'checkpoints':        'E:/Brain Tumer/outputs/checkpoints',
        'logs':               'E:/Brain Tumer/outputs/logs',
        'plots':              'E:/Brain Tumer/outputs/plots',
        'tables':             'E:/Brain Tumer/outputs/results/tables',
    },
    'data': {
        'image_size': 224,
        'image_channels': 3,
        'max_slices_per_patient': 30,
        'train_ratio': 0.70,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'stratify_by': 'IDH1',
        'clinical_features': [
            'Gender', 'Age_at_scan_years',
            'Survival_from_surgery_days_UPDATED',
            'IDH1', 'MGMT', 'GTR_over90percent', 'PsP_TP_score'
        ],
    },
    'model': {
        'backbone': 'swin_tiny_patch4_window7_224',
        'pretrained': True,
        'embed_dim': 768,
        'num_attention_heads': 8,
        'fusion_dim': 256,
        'temporal_depth': 2,
        'dropout': 0.1,
    },
    'tasks': {
        'segmentation': True,
        'captioning': True,
        'grounding': True,
        'num_seg_classes': 2,
        'num_grounding_queries': 10,
    },
    'training': {
        'epochs': 50,
        'batch_size': 2,
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'scheduler': 'cosine',
        'warmup_epochs': 5,
        'grad_clip': 1.0,
        'mixed_precision': False,
        'save_best_only': True,
        'task_weights': {
            'segmentation': 1.0,
            'captioning': 0.5,
            'grounding': 0.5
        },
    },
    'few_shot': {'enabled': True, 'k_shots': [1, 5, 10]},
    'zero_shot': {'enabled': True},
    'evaluation': {
        'metrics': [
            'accuracy', 'precision', 'recall', 'f1',
            'auc_roc', 'confusion_matrix', 'dice',
            'iou', 'bleu', 'rouge', 'map'
        ],
        'statistical_tests': ['wilcoxon', 'mcnemar', 'delong', 'kaplan_meier'],
        'save_plots': True,
        'save_tables': True,
    },
}

pathlib.Path('configs').mkdir(exist_ok=True)
with open('configs/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
print('Config written OK')

# ── Verify all paths ──────────────────────────
print('\n--- Path Verification ---')
for key, val in cfg['paths'].items():
    if val == '':
        print(f'  {key}: (empty - OK)')
        continue
    p = pathlib.Path(val)
    exists = p.exists()
    print(f'  {key}: {exists} → {val}')

# ── List Dataset folder ───────────────────────
ds = pathlib.Path('E:/Brain Tumer/Dataset')
print('\n--- Dataset folder contents ---')
if ds.exists():
    for item in ds.iterdir():
        print(f'  {item.name}')
else:
    print('  Dataset folder NOT found!')

# ── Try reading CSV ───────────────────────────
print('\n--- CSV Test ---')
csv_path = pathlib.Path('E:/Brain Tumer/Dataset/UPENN-GBM_clinical_info_v2_1.csv')
if csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f'  CSV loaded OK: {df.shape[0]} rows, {df.shape[1]} cols')
    print(f'  Columns: {list(df.columns)}')
else:
    print('  CSV NOT found - listing all files in Dataset:')
    for f in ds.rglob('*.csv'):
        print(f'    {f}')