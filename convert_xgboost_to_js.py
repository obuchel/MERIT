"""
convert_xgboost_to_js.py
========================
Converts XGBoost models stored in a pickle file to the compact JavaScript
tree format used by xgboost_trees_data.js.

Output format (mirrors the existing JS file):
  export const XGBOOST_TREES = {
    "<model_name>": {
      "b": <base_score>,          // bias / intercept
      "f": ["feat0", "feat1", ...],  // feature names (index = column index)
      "t": [                      // list of trees
        [                         // one tree = flat array of nodes
          [featIdx, threshold, yesChild, noChild],  // split node
          [leafValue],                               // leaf node
          ...
        ],
        ...
      ]
    },
    ...
  };

Usage:
  python convert_xgboost_to_js.py \
      --input  rfp_models_v5.pkl \
      --output xgboost_trees_data.js \
      [--precision 6]             # decimal places for floats (default: 6)
"""

import argparse
import json
import pickle
import re
import sys


# ---------------------------------------------------------------------------
# Tree parsing
# ---------------------------------------------------------------------------

def parse_tree(tree_str: str, feature_names: list[str]) -> list:
    """
    Convert one XGBoost text-dump tree into a flat list of nodes.

    Each node is either:
      [featureIdx, threshold, yesChildIdx, noChildIdx]   – internal split
      [leafValue]                                         – leaf

    Nodes are stored in ascending node-id order so child indices can be used
    directly as list indices.
    """
    feat_index = {name: i for i, name in enumerate(feature_names)}
    nodes: dict[int, list] = {}

    for raw_line in tree_str.strip().splitlines():
        line = raw_line.strip()

        # --- leaf ---------------------------------------------------------
        leaf_m = re.match(r'(\d+):leaf=([^\s,]+)', line)
        if leaf_m:
            nid  = int(leaf_m.group(1))
            val  = float(leaf_m.group(2))
            nodes[nid] = [val]
            continue

        # --- split --------------------------------------------------------
        # Pattern: N:[feat<thresh] yes=Y,no=N,missing=M
        split_m = re.match(
            r'(\d+):\[(.+?)<(.+?)\]\s+yes=(\d+),no=(\d+)', line
        )
        if split_m:
            nid    = int(split_m.group(1))
            feat   = split_m.group(2)
            thresh = float(split_m.group(3))
            yes    = int(split_m.group(4))
            no     = int(split_m.group(5))

            # Resolve feature to integer index
            if feat.startswith('f') and feat[1:].isdigit():
                fidx = int(feat[1:])          # f0, f1, … (anonymous features)
            else:
                fidx = feat_index.get(feat)
                if fidx is None:
                    raise ValueError(
                        f"Feature '{feat}' not found in feature list. "
                        f"Available: {feature_names[:5]} …"
                    )

            nodes[nid] = [fidx, thresh, yes, no]

    if not nodes:
        return []

    max_nid = max(nodes)
    # Fill any gaps with a zero leaf (should not happen in well-formed dumps)
    return [nodes.get(i, [0.0]) for i in range(max_nid + 1)]


def round_node(node: list, precision: int) -> list:
    """Round all floats in a node to *precision* decimal places."""
    return [round(v, precision) if isinstance(v, float) else v for v in node]


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------

def extract_base_score(booster) -> float:
    """
    Extract the base (bias) score from a Booster object.
    Works across different XGBoost versions.
    """
    try:
        config = json.loads(booster.save_config())
        raw = config['learner']['learner_model_param']['base_score']
        return float(raw)
    except Exception:
        pass
    # Fallback: parse from text model dump header
    try:
        model_str = booster.save_raw(raw_format='ubj')  # binary
    except Exception:
        pass
    return 0.5  # XGBoost default


def convert_pickle(
    input_path: str,
    output_path: str,
    precision: int = 6,
    var_name: str = "XGBOOST_TREES",
) -> None:
    """
    Load a pickle that contains XGBoost models and write a JS file.

    Supported pickle layouts:
      1. dict with keys 'models' (dict of XGBRegressor/XGBClassifier) and
         'feature_sets' (dict of feature-name lists) — the v5 layout.
      2. dict mapping model-name → XGBModel directly.
      3. A single XGBModel (exported as "model").
    """
    print(f"Loading pickle: {input_path}")
    with open(input_path, 'rb') as fh:
        raw = pickle.load(fh)

    # --- resolve models + feature_sets ------------------------------------
    if isinstance(raw, dict) and 'models' in raw:
        # Layout 1: {'models': {...}, 'feature_sets': {...}, ...}
        model_dict   = raw['models']
        feature_sets = raw.get('feature_sets', {})
    elif isinstance(raw, dict):
        # Layout 2: {name: XGBModel, ...}
        model_dict   = raw
        feature_sets = {}
    else:
        # Layout 3: single model
        model_dict   = {'model': raw}
        feature_sets = {}

    output: dict[str, dict] = {}

    for model_name, model in model_dict.items():
        if not hasattr(model, 'get_booster'):
            print(f"  Skipping '{model_name}' (not an XGBoost model)")
            continue

        booster      = model.get_booster()
        feature_names = feature_sets.get(model_name, [])

        # Fall back to booster's own feature names if not in feature_sets
        if not feature_names:
            feature_names = booster.feature_names or []

        base_score   = extract_base_score(booster)
        raw_trees    = booster.get_dump(with_stats=False)

        print(
            f"  {model_name}: {len(raw_trees)} trees, "
            f"{len(feature_names)} features, base={base_score:.6f}"
        )

        trees = []
        for i, tree_str in enumerate(raw_trees):
            try:
                nodes = parse_tree(tree_str, feature_names)
                nodes = [round_node(n, precision) for n in nodes]
                trees.append(nodes)
            except Exception as exc:
                print(f"    WARNING: could not parse tree {i}: {exc}")
                trees.append([])

        output[model_name] = {
            'b': round(base_score, precision),
            'f': feature_names,
            't': trees,
        }

    # --- serialise to compact JSON / JS -----------------------------------
    # Use compact JSON (no extra whitespace) to mirror the example file style
    json_str = json.dumps(output, separators=(',', ':'))

    # Build model summary comment
    model_summary = ', '.join(
        f"{name}({len(v['t'])} trees)" for name, v in output.items()
    )
    header = (
        f"// {output_path.split('/')[-1]}\n"
        f"// Auto-generated from {input_path.split('/')[-1]}\n"
        f"// {len(output)} XGBRegressor models: {model_summary}\n\n"
        f"export const {var_name} = {json_str};\n"
    )

    with open(output_path, 'w') as fh:
        fh.write(header)

    # Quick size report
    size_kb = len(header.encode()) / 1024
    print(f"\nWrote {output_path}  ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Convert XGBoost pickle to compact JavaScript tree format.'
    )
    parser.add_argument('--input',  '-i', required=True,
                        help='Path to the .pkl file')
    parser.add_argument('--output', '-o', required=True,
                        help='Path for the output .js file')
    parser.add_argument('--precision', '-p', type=int, default=6,
                        help='Decimal places for floats (default: 6)')
    parser.add_argument('--var-name', default='XGBOOST_TREES',
                        help='JS export variable name (default: XGBOOST_TREES)')
    args = parser.parse_args()

    convert_pickle(
        input_path  = args.input,
        output_path = args.output,
        precision   = args.precision,
        var_name    = args.var_name,
    )


if __name__ == '__main__':
    main()
