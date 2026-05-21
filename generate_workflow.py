import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(22, 30), facecolor='#f8f9fa')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 22)
ax.set_ylim(0, 30)
ax.axis('off')

BLUE   = '#1565C0'
DBLUE  = '#0D47A1'
GREEN  = '#2E7D32'
ORANGE = '#E65100'
PURPLE = '#6A1B9A'
RED    = '#B71C1C'
TEAL   = '#00695C'
GRAY   = '#37474F'
LGRAY  = '#90A4AE'
AMBER  = '#F57F17'
PINK   = '#AD1457'


def box(ax, x, y, w, h, text, color, textcolor='white', fontsize=9, bold=False, subtext=None):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.08', linewidth=1.2,
        edgecolor='#333', facecolor=color, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    yoff = 0.15 if subtext else 0
    ax.text(x, y + yoff, text, ha='center', va='center', fontsize=fontsize,
        color=textcolor, fontweight=weight, zorder=4, multialignment='center')
    if subtext:
        ax.text(x, y - 0.22, subtext, ha='center', va='center', fontsize=7,
            color=textcolor, alpha=0.85, zorder=4, multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#444', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=2)


def section_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=11, fontweight='bold', color=color,
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15,
                  edgecolor=color, lw=1.5))


# ── TITLE ──
ax.text(11, 29.4, 'Froth CNN Training Tool — End-to-End Workflow',
    ha='center', va='center', fontsize=16, fontweight='bold', color='#1a1a2e',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#1a1a2e', lw=2))

# ══════════════════════════════════════════════
# 1  DATA SOURCES
# ══════════════════════════════════════════════
section_label(ax, 11, 28.6, '① DATA SOURCES', BLUE)

box(ax, 4,   27.8, 3.4, 0.9, 'forms/D01-D17.xlsx', BLUE,
    fontsize=8, bold=True,
    subtext='Kinetik_Numuneler D8:D14\nPb Tenoru (%) per time frame')
box(ax, 9.5, 27.8, 3.4, 0.9, 'frames/D01-D17/Y1-Y7/', BLUE,
    fontsize=8, bold=True,
    subtext='PNG images per experiment\nper time frame (~32 imgs/folder)')
box(ax, 15,  27.8, 3.4, 0.9, 'Experiment Details', BLUE,
    fontsize=8, bold=True,
    subtext='Name, date, operator,\nchemical conditions')

box(ax, 11, 26.7, 5.2, 0.75, 'Auto-Import  (Experiments Panel)', DBLUE,
    fontsize=8, bold=True,
    subtext='Parses all xlsx forms + maps Y1-Y7 folders automatically')

arrow(ax, 4,   27.35, 8.4,  26.7, BLUE)
arrow(ax, 9.5, 27.35, 10.5, 27.07, BLUE)
arrow(ax, 15,  27.35, 13.6, 26.7, BLUE)
arrow(ax, 11, 26.32, 11, 25.85, DBLUE)

# ══════════════════════════════════════════════
# 2  SPLIT ASSIGNMENT
# ══════════════════════════════════════════════
section_label(ax, 11, 25.55, '② SPLIT ASSIGNMENT  (per experiment, not per image)', TEAL)

box(ax, 5,    24.7, 3.0, 0.9, 'TRAIN\n(e.g. D01-D15)', GREEN,
    fontsize=8, bold=True, subtext='Weight updates\n& early stopping')
box(ax, 11,   24.7, 3.0, 0.9, 'TEST\n(e.g. D16)', AMBER, '#333',
    fontsize=8, bold=True, subtext='Eval each epoch\nfor model selection')
box(ax, 17,   24.7, 3.0, 0.9, 'VALIDATION\n(e.g. D17)', RED,
    fontsize=8, bold=True, subtext='HELD OUT\nevaluated ONCE only')

ax.text(17, 24.05,
    'WARNING: NEVER used during\ntraining or grid search',
    ha='center', fontsize=7.5, color=RED, fontweight='bold',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor=RED, lw=1))

for xd in [5, 11, 17]:
    arrow(ax, 11, 25.2, xd, 25.15, TEAL)

# ══════════════════════════════════════════════
# 3  PREPROCESSING
# ══════════════════════════════════════════════
section_label(ax, 11, 23.4, '③ PREPROCESSING  (Preprocessing Tab)', PURPLE)

ax.text(5, 23.1, 'TRAIN pipeline', ha='center', fontsize=8, color=GREEN, fontweight='bold')
train_steps = [(5, 22.6), (5, 21.9), (5, 21.2), (5, 20.5)]
train_labels = ['Crop (optional)', 'Resize -> 224x224', 'Augmentation\n(random each epoch)', 'ToTensor + Normalize']
train_colors = ['#7B1FA2', '#7B1FA2', '#4A148C', '#7B1FA2']
for (x, y), lbl, c in zip(train_steps, train_labels, train_colors):
    box(ax, x, y, 2.8, 0.58, lbl, c, fontsize=8)
for i in range(len(train_steps) - 1):
    arrow(ax, train_steps[i][0], train_steps[i][1] - 0.29,
          train_steps[i+1][0], train_steps[i+1][1] + 0.29, PURPLE)

ax.text(8.5, 21.2,
    'H/V flip  |  Rotation +-deg\nColor jitter  |  Gaussian blur\nRandom erasing  |  Resized crop',
    ha='left', va='center', fontsize=7, color='#4A148C',
    bbox=dict(boxstyle='round,pad=0.25', facecolor='#EDE7F6', edgecolor='#7B1FA2', lw=1))
ax.annotate('', xy=(8.45, 21.2), xytext=(6.85, 21.2),
    arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.2))

ax.text(14, 23.1, 'TEST / VAL pipeline', ha='center', fontsize=8, color=ORANGE, fontweight='bold')
eval_steps = [(14, 22.6), (14, 21.9), (14, 21.2), (14, 20.5)]
eval_labels = ['Crop (optional)', 'Resize -> 224x224', 'NO augmentation\n(deterministic)', 'ToTensor + Normalize']
eval_colors = ['#E65100', '#E65100', '#BF360C', '#E65100']
for (x, y), lbl, c in zip(eval_steps, eval_labels, eval_colors):
    box(ax, x, y, 2.8, 0.58, lbl, c, fontsize=8)
for i in range(len(eval_steps) - 1):
    arrow(ax, eval_steps[i][0], eval_steps[i][1] - 0.29,
          eval_steps[i+1][0], eval_steps[i+1][1] + 0.29, ORANGE)

ax.text(16.9, 21.2, 'Ensures fair, reproducible\neval numbers',
    ha='left', va='center', fontsize=7, color='#BF360C',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FBE9E7', edgecolor=ORANGE, lw=1))

arrow(ax, 5,  24.25, 5,  23.28, GREEN)
arrow(ax, 11, 24.25, 14, 23.28, ORANGE)
arrow(ax, 17, 24.25, 14, 23.28, RED)

for y in [22.6, 21.9, 21.2, 20.5]:
    ax.plot([7.15, 12.85], [y, y], color=LGRAY, lw=0.7, linestyle=':', zorder=1)

arrow(ax, 5,  20.21, 5,  19.7,  PURPLE)
arrow(ax, 14, 20.21, 14, 19.7,  ORANGE)

# ══════════════════════════════════════════════
# 4  MODEL BUILDING
# ══════════════════════════════════════════════
section_label(ax, 11, 19.4, '④ MODEL  (Model Tab)', GRAY)

box(ax, 5.5, 18.75, 3.4, 0.82, 'Transfer Learning', GRAY,
    fontsize=8, bold=True,
    subtext='Pretrained backbone (ImageNet)\n+ new regression head')
box(ax, 15.5, 18.75, 3.4, 0.82, 'From Scratch', GRAY,
    fontsize=8, bold=True,
    subtext='N conv blocks x base_filters\nFC layers -> output(1)')

arrow(ax, 9.5,  19.15, 7.2,  18.95, GRAY)
arrow(ax, 12.5, 19.15, 13.8, 18.95, GRAY)

ax.text(3.2, 18.05,
    'Freeze backbone:\n  Only head trains (fast,\n  good for small datasets)\n\nUnfreeze last N:\n  Fine-tune top N layers\n  with low learning rate',
    ha='left', va='center', fontsize=7.5, color='#333',
    bbox=dict(boxstyle='round,pad=0.28', facecolor='#ECEFF1', edgecolor=GRAY, lw=1))

ax.text(8.8, 18.1,
    'Regression Head:  GlobalAvgPool -> Flatten -> Dropout -> Linear(in_features, 1)\n'
    'Output: single float = predicted Pb Tenoru (%)\n'
    'No activation on output (pure regression)',
    ha='left', va='center', fontsize=8, color='white',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#37474F', edgecolor='#90A4AE', lw=1.5))

arrow(ax, 11, 18.34, 11, 17.82, GRAY)

# ══════════════════════════════════════════════
# 5  TRAINING LOOP
# ══════════════════════════════════════════════
section_label(ax, 11, 17.52, '⑤ TRAINING LOOP  (Training Tab)', GREEN)

loop_box = FancyBboxPatch((1.4, 13.5), 19.2, 3.72,
    boxstyle='round,pad=0.1', linewidth=2,
    edgecolor=GREEN, facecolor='#F1F8E9', zorder=1, linestyle='--')
ax.add_patch(loop_box)
ax.text(2.0, 17.1, 'for epoch in 1..epochs:', fontsize=8.5,
    color=GREEN, fontweight='bold', va='top')

box(ax, 5,   16.5, 3.0, 0.72, 'Forward Pass\n(train batch)', '#1B5E20', fontsize=8)
box(ax, 9.2, 16.5, 2.7, 0.72, 'Loss\nMSE / MAE / Huber', '#1B5E20', fontsize=8)
box(ax, 13.2,16.5, 2.8, 0.72, 'Backprop +\nOptimizer Step', '#1B5E20', fontsize=8)
box(ax, 17.5,16.5, 2.6, 0.72, 'LR Scheduler\nStep', '#388E3C', fontsize=8)

arrow(ax, 6.5,  16.5, 7.85, 16.5, GREEN)
arrow(ax, 10.55,16.5, 11.8, 16.5, GREEN)
arrow(ax, 14.6, 16.5, 16.2, 16.5, GREEN)

box(ax, 5,   15.1, 3.0, 0.72, 'Eval on TEST set\n(no grad, no augment)', AMBER, '#333', fontsize=8)
box(ax, 9.2, 15.1, 2.7, 0.72, 'Metrics:\nRMSE, MAE, R2', AMBER, '#333', fontsize=8)
box(ax, 13.2,15.1, 2.8, 0.72, 'Early Stopping\nCheck', '#E65100', fontsize=8)
box(ax, 17.5,15.1, 2.6, 0.72, 'Save Best\nCheckpoint (.pt)', '#2E7D32', fontsize=8)

arrow(ax, 5,    16.14, 5,    15.46, AMBER)
arrow(ax, 6.5,  15.1,  7.85, 15.1,  AMBER)
arrow(ax, 10.55,15.1,  11.8, 15.1,  AMBER)
arrow(ax, 14.6, 15.1,  16.2, 15.1,  AMBER)

ax.text(13.2, 14.52,
    'if no improvement >=\npatience epochs -> stop',
    ha='center', fontsize=7, color='#E65100',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF3E0', edgecolor='#E65100', lw=1))

ax.annotate('', xy=(3.0, 16.5), xytext=(3.0, 14.75),
    arrowprops=dict(arrowstyle='<-', color=GREEN, lw=1.5))
ax.text(2.3, 15.6, 'next\nbatch', ha='center', fontsize=7.5, color=GREEN)

arrow(ax, 11, 17.16, 11, 17.22, GREEN)
arrow(ax, 11, 13.5,  11, 13.02, GREEN)

# ══════════════════════════════════════════════
# 6  GRID SEARCH WRAPPER
# ══════════════════════════════════════════════
section_label(ax, 11, 12.72, '⑥ GRID SEARCH  (optional wrapper around step 5)', PINK)

gs_box = FancyBboxPatch((1.4, 11.35), 19.2, 1.12,
    boxstyle='round,pad=0.1', linewidth=1.5,
    edgecolor=PINK, facecolor='#FCE4EC', zorder=1, linestyle='--')
ax.add_patch(gs_box)

ax.text(2.0, 12.32,
    'Cartesian product of all multi-valued params:'
    '  e.g.  batch=[16,32] x lr=[0.001,0.0001] x opt=[Adam,SGD]  ->  8 runs',
    ha='left', va='center', fontsize=8, color=PINK, fontweight='bold')
ax.text(2.0, 11.78,
    'Each run = one full independent training cycle  |  '
    'Best run selected by lowest TEST RMSE  |  '
    'Validation is NOT touched here',
    ha='left', va='center', fontsize=8, color='#880E4F')

arrow(ax, 11, 11.35, 11, 10.82, PINK)

# ══════════════════════════════════════════════
# 7  FINAL VALIDATION
# ══════════════════════════════════════════════
section_label(ax, 11, 10.52, '⑦ FINAL VALIDATION  (once, ever)', RED)

box(ax, 7,   9.82, 4.2, 0.78, 'Load best checkpoint\n(lowest test loss)', GREEN, fontsize=8)
box(ax, 14.5,9.82, 4.8, 0.78, 'Evaluate on VALIDATION set\nRMSE, MAE, R2 for final report', RED, fontsize=8)
arrow(ax, 9.1, 9.82, 12.1, 9.82, RED)
arrow(ax, 11, 10.16, 11, 10.21, RED)

ax.text(14.5, 9.22,
    'Only honest estimate of real-world performance.\nDo not use this number to pick hyperparameters.',
    ha='center', fontsize=7.5, color=RED, style='italic',
    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE', edgecolor=RED, lw=1))

arrow(ax, 11, 9.43, 11, 8.92, GRAY)

# ══════════════════════════════════════════════
# 8  RESULTS
# ══════════════════════════════════════════════
section_label(ax, 11, 8.62, '⑧ RESULTS  (Results Tab)', TEAL)

box(ax, 4.5,  7.82, 3.4, 0.88, 'Results Table', TEAL,
    fontsize=8, bold=True,
    subtext='All runs: params, test RMSE\nMAE, R2, training time')
box(ax, 11,   7.82, 3.4, 0.88, 'Loss Curves', TEAL,
    fontsize=8, bold=True,
    subtext='Train + test loss per epoch\nBest epoch marked')
box(ax, 17.5, 7.82, 3.4, 0.88, 'Metric Charts', TEAL,
    fontsize=8, bold=True,
    subtext='Test vs Val bar chart\nper selected run')

for xd in [4.5, 11, 17.5]:
    arrow(ax, 11, 8.21, xd, 8.26, TEAL)

# ══════════════════════════════════════════════
# BOTTOM REFERENCE TABLE
# ══════════════════════════════════════════════
ax.add_patch(FancyBboxPatch((0.3, 0.25), 21.4, 7.0,
    boxstyle='round,pad=0.1', linewidth=1.5,
    edgecolor='#9E9E9E', facecolor='#FAFAFA', zorder=0))

ax.text(11, 7.08,
    'PARAMETER REFERENCE  +  COMMON QUESTIONS',
    ha='center', fontsize=11, fontweight='bold', color='#212121')

# Q&A section
qa_texts = [
    ('Q: Center crop 224 then resize 224 — any issue?',
     'No crash, but resize is a no-op (wasted step). Best practice: crop to remove borders '
     '(e.g. crop 400x400 to cut reactor walls), THEN resize to 224. '
     'If crop is SMALLER than resize target the image is upscaled — avoidable quality loss.'),
    ('Q: How does Random Rotation work?',
     'Picks angle in [-degrees, +degrees] uniformly at random for EACH image load during training. '
     'Rotates with bilinear interpolation. '
     'Empty corners are filled with black. '
     'Good for froth (no inherent orientation). Keep <=30 deg to avoid large black corners.'),
]
yqa = 6.72
for q, a in qa_texts:
    ax.text(0.7, yqa, q, fontsize=8, fontweight='bold', color=PURPLE, va='top')
    yqa -= 0.28
    ax.text(0.7, yqa, a, fontsize=7.5, color='#333', va='top', wrap=True)
    yqa -= 0.52

# Parameter table
headers = ['PARAMETER', 'WHAT IT DOES', 'GOOD DEFAULT']
col_x   = [0.55, 3.85, 17.55]
col_w2  = [3.1,  13.5, 3.6]
row_h2  = 0.2
y_header = yqa - 0.02

for ci, (hdr, cw) in enumerate(zip(headers, col_w2)):
    ax.add_patch(FancyBboxPatch((col_x[ci], y_header - row_h2), cw, row_h2,
        boxstyle='square,pad=0', facecolor='#37474F', edgecolor='white', lw=0.5, zorder=2))
    ax.text(col_x[ci] + cw/2, y_header - row_h2/2, hdr,
        ha='center', va='center', fontsize=7.5, color='white', fontweight='bold', zorder=3)

params_table = [
    ('Batch size',        'Images per gradient update. Larger = stable gradients, needs more VRAM. Too large may generalise worse.', '32'),
    ('Learning rate',     'Step size for each weight update. Too high = training diverges. Too low = very slow convergence.', '0.001'),
    ('Optimizer: Adam',   'Adaptive per-parameter LR. Momentum + RMS scaling. Best general-purpose choice.', 'use first'),
    ('Optimizer: SGD',    'Classic gradient descent + momentum. Slower but can generalise better with tuning.', 'momentum=0.9'),
    ('Optimizer: AdamW',  'Adam with decoupled weight decay. Better regularisation than Adam.', 'wd=1e-4'),
    ('Weight decay',      'L2 penalty added to loss. Shrinks weights toward zero. Prevents overfitting.', '1e-4'),
    ('Loss: MSE',         'Mean Squared Error. Squares errors so large mistakes are penalised hard. Standard for regression.', 'default'),
    ('Loss: MAE',         'Mean Absolute Error. Linear penalty. More robust when some Pb values are outliers.', 'if outliers'),
    ('Loss: Huber',       'MSE when |error|<delta, MAE otherwise. Combines stability and robustness.', 'good alt'),
    ('StepLR',            'Multiply LR by gamma every step_size epochs. Simple, predictable decay.', 'step=10, g=0.5'),
    ('CosineAnnealing',   'Smoothly lowers LR following a cosine curve over T_max epochs. No jumps.', 'T_max=epochs'),
    ('ReduceOnPlateau',   'Halves LR when test loss stops improving for patience epochs. Reactive.', 'patience=5'),
    ('Early stopping',    'Halts training when test loss has not improved by min_delta for patience consecutive epochs. Loads best weights.', 'patience=15'),
    ('Freeze backbone',   'Locks all pretrained conv weights; only the regression head is trained. Fast, safe for small datasets.', 'try first'),
    ('Dropout',           'Randomly zeros p fraction of neurons each forward pass during training. Reduces co-adaptation / overfitting.', '0.5'),
]

for ri, (p, d, dflt) in enumerate(params_table):
    yy = y_header - (ri + 2) * row_h2
    bg = '#F5F5F5' if ri % 2 == 0 else 'white'
    for ci, cw in enumerate(col_w2):
        ax.add_patch(FancyBboxPatch((col_x[ci], yy), cw, row_h2,
            boxstyle='square,pad=0', facecolor=bg, edgecolor='#E0E0E0', lw=0.4, zorder=2))
    vals = [p, d, dflt]
    vcolors = [DBLUE, '#333', GREEN]
    for ci, (cw, val, vc) in enumerate(zip(col_w2, vals, vcolors)):
        ax.text(col_x[ci] + cw/2, yy + row_h2/2, val,
            ha='center', va='center', fontsize=6.8, color=vc, zorder=3, multialignment='center')

plt.savefig('workflow.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print('Saved workflow.png')
