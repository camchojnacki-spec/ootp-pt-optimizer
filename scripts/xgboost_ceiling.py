"""XGBoost ceiling check — how much signal is left in the rating features?

Fits a gradient-boosted model on every available numeric + categorical
feature and compares CV R² to the linear meta correlation. If XGBoost
tops out close to our current r, the linear model is near the ceiling.
If it blows past us, non-linear interactions are still available to
capture.
"""
import sys, sqlite3
sys.path.insert(0, '.')
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from app.core.database import get_db_path

POS_COLS = ['pos_rating_c', 'pos_rating_1b', 'pos_rating_2b', 'pos_rating_3b',
            'pos_rating_ss', 'pos_rating_lf', 'pos_rating_cf', 'pos_rating_rf']

conn = sqlite3.connect(get_db_path())
conn.row_factory = sqlite3.Row


def _to_float(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    if '-' in s:
        try:
            a, b = s.split('-')[:2]
            return (float(a) + float(b)) / 2.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def evaluate(label: str, rows, feature_names, target_values):
    X = np.array([[_to_float(r[f]) for f in feature_names] for r in rows], dtype=float)
    y = np.array(target_values, dtype=float)
    # Drop rows with any NaN target
    keep = ~np.isnan(y)
    X = X[keep]; y = y[keep]
    # Replace NaN features with col median
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            X[np.isnan(col), j] = 0 if np.isnan(med) else med
    print(f'\n{label}: n={len(X)}, features={len(feature_names)}')

    # 1. XGBoost CV R²
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        tree_method='hist', verbosity=0,
    )
    cv = cross_val_score(model, X, y, cv=5, scoring='r2')
    xgb_cv_r2 = float(np.mean(cv))
    model.fit(X, y)
    preds = model.predict(X)
    xgb_in_sample_r = float(np.corrcoef(preds, y)[0, 1])
    print(f'  XGBoost: CV R² = {xgb_cv_r2:+.4f}  (in-sample r = {xgb_in_sample_r:+.4f})')

    # 2. Feature importances (top 10)
    importances = sorted(zip(feature_names, model.feature_importances_),
                         key=lambda x: -x[1])[:10]
    print(f'  Top 10 features by importance:')
    for f, imp in importances:
        print(f'    {f:30s}  {imp:.4f}')


# ── Batting ──
bat_rows = conn.execute(f"""
    SELECT c.card_id, c.meta_score_batting m, c.card_value ovr, c.card_type,
           c.age,
           c.contact, c.gap_power, c.power, c.eye, c.avoid_ks, c.babip,
           c.contact_vl, c.contact_vr, c.power_vl, c.power_vr, c.eye_vl, c.eye_vr,
           c.gap_vl, c.gap_vr, c.avoid_ks_vl, c.avoid_ks_vr,
           c.speed, c.stealing, c.baserunning,
           {','.join('c.' + x for x in POS_COLS)},
           c.position,
           c.infield_range, c.infield_error, c.infield_arm,
           c.catcher_ability, c.catcher_frame, c.catcher_arm,
           c.of_range, c.of_error, c.of_arm,
           bs.war, bs.pa
    FROM cards c JOIN batting_stats bs ON bs.card_id = c.card_id
    JOIN (SELECT card_id, MAX(snapshot_date) mx FROM batting_stats
          WHERE card_id IS NOT NULL GROUP BY card_id, league_id) l
      ON bs.card_id = l.card_id AND bs.snapshot_date = l.mx
    WHERE bs.pa >= 150 AND c.card_value > 0 AND bs.war IS NOT NULL
""").fetchall()

bat_features = [
    'ovr', 'card_type', 'age', 'position',
    'contact', 'gap_power', 'power', 'eye', 'avoid_ks', 'babip',
    'contact_vl', 'contact_vr', 'power_vl', 'power_vr', 'eye_vl', 'eye_vr',
    'gap_vl', 'gap_vr', 'avoid_ks_vl', 'avoid_ks_vr',
    'speed', 'stealing', 'baserunning',
    *POS_COLS,
    'infield_range', 'infield_error', 'infield_arm',
    'catcher_ability', 'catcher_frame', 'catcher_arm',
    'of_range', 'of_error', 'of_arm',
]
bat_targets = [r['war'] * 600.0 / r['pa'] for r in bat_rows]
evaluate('BATTING WAR/600', bat_rows, bat_features, bat_targets)

# Also pure-linear benchmark: current meta correlation (filter NaN both sides)
bat_metas = np.array([_to_float(r['m']) for r in bat_rows])
bat_targs = np.array([_to_float(t) for t in bat_targets])
keep = ~(np.isnan(bat_metas) | np.isnan(bat_targs))
linear_r = float(np.corrcoef(bat_metas[keep], bat_targs[keep])[0, 1])
print(f'  Current linear meta r:  {linear_r:+.4f}  (in-sample, all rows)')


# ── Pitching ──
pit_rows = conn.execute("""
    SELECT c.card_id, c.meta_score_pitching m, c.card_value ovr, c.card_type,
           c.age,
           c.stuff, c.movement, c.control, c.p_hr, c.p_babip,
           c.stuff_vl, c.stuff_vr, c.movement_vl, c.movement_vr,
           c.control_vl, c.control_vr, c.p_hr_vl, c.p_hr_vr,
           c.stamina, c.hold, c.velocity, c.pitcher_role,
           ps.war, ps.ip
    FROM cards c JOIN pitching_stats ps ON ps.card_id = c.card_id
    JOIN (SELECT card_id, MAX(snapshot_date) mx FROM pitching_stats
          WHERE card_id IS NOT NULL GROUP BY card_id, league_id) l
      ON ps.card_id = l.card_id AND ps.snapshot_date = l.mx
    WHERE ps.ip >= 30 AND c.card_value > 0 AND ps.war IS NOT NULL
""").fetchall()

pit_features = [
    'ovr', 'card_type', 'age', 'pitcher_role',
    'stuff', 'movement', 'control', 'p_hr', 'p_babip',
    'stuff_vl', 'stuff_vr', 'movement_vl', 'movement_vr',
    'control_vl', 'control_vr', 'p_hr_vl', 'p_hr_vr',
    'stamina', 'hold', 'velocity',
]
pit_targets = [r['war'] * 200.0 / r['ip'] for r in pit_rows]
evaluate('PITCHING WAR/200', pit_rows, pit_features, pit_targets)
pit_metas = np.array([_to_float(r['m']) for r in pit_rows])
pit_targs = np.array([_to_float(t) for t in pit_targets])
keep = ~(np.isnan(pit_metas) | np.isnan(pit_targs))
linear_r = float(np.corrcoef(pit_metas[keep], pit_targs[keep])[0, 1])
print(f'  Current linear meta r:  {linear_r:+.4f}  (in-sample, all rows)')

# Also compute LINEAR CV R² (comparable to XGBoost CV R²) for a fair benchmark
from sklearn.linear_model import LinearRegression
print()
print('── Linear vs XGBoost CV R² (apples-to-apples comparison) ──')
for name, X_rows, feats, ys in [('BATTING', bat_rows, bat_features, bat_targets),
                                 ('PITCHING', pit_rows, pit_features, pit_targets)]:
    X = np.array([[_to_float(r[f]) for f in feats] for r in X_rows], dtype=float)
    y = np.array(ys, dtype=float)
    keep = ~np.isnan(y)
    X = X[keep]; y = y[keep]
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.isnan(col).any():
            med = np.nanmedian(col)
            X[np.isnan(col), j] = 0 if np.isnan(med) else med
    lin_cv = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
    print(f'  {name:10s}  Linear CV R² = {float(np.mean(lin_cv)):+.4f}')
