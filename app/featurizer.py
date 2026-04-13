import pandas as pd
import numpy as np
import re
from math import gcd
from functools import reduce


def parse_formula(formula):
    pattern = r'([A-Z][a-z]*)(\d*\.?\d*)'
    matches = re.findall(pattern, formula)
    composition = {}
    for (element, count) in matches:
        count = float(count) if count != '' else 1.0
        composition[element] = composition.get(element, 0) + count
    return composition


def calculate_gcf(numbers):
    return reduce(gcd, [int(num) for num in numbers])


def calculate_features(material_dict, elements_df):
    features = {}
    counts = list(material_dict.values())
    try:
        gcf = calculate_gcf(counts)
    except:
        gcf = 1

    for prop in elements_df.columns[1:]:
        values = []
        for element, count in material_dict.items():
            if element in elements_df['Elements'].values:
                # FIXED THE NAME ERROR HERE
                value = elements_df.loc[elements_df['Elements'] == element, prop].values[0]
                values.extend([value] * int(count / gcf))
            else:
                values = [np.nan]
                break

        prop_values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        if prop_values:
            features[f'sum_{prop}'] = sum(prop_values)
            features[f'mean_{prop}'] = np.mean(prop_values)
            features[f'std_dev_{prop}'] = np.std(prop_values)
            features[f'max_{prop}'] = max(prop_values)
            features[f'min_{prop}'] = min(prop_values)
            features[f'diff_{prop}'] = max(prop_values) - min(prop_values)
    return features


def main():
    print("🚀 Starting Featurizer...")
    try:
        df_new = pd.read_csv('material_band_gap.csv')
        # Added latin1 encoding just in case of the Å symbol we saw earlier
        df_elements = pd.read_csv('elemental_properties.csv')

        for col in df_elements.columns[1:]:
            df_elements[col] = pd.to_numeric(df_elements[col], errors='coerce')

        features_list = []
        for index, row in df_new.iterrows():
            if index % 500 == 0:
                print(f"Processing: {index}/{len(df_new)}")

            material_dict = parse_formula(str(row['Compounds']))
            features = calculate_features(material_dict, df_elements)
            features_list.append(features)

        df_features = pd.DataFrame(features_list)
        df_combined = pd.concat([df_new, df_features], axis=1)
        df_combined.to_csv('processed_features_v2.csv', index=False)
        print("✅ Success: processed_features_v2.csv created.")

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file. {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def get_single_feature_vector(formula, elements_df):
    """
    Wrapper function for the API to convert a string formula
    directly into a DataFrame that the model can read.
    """
    material_dict = parse_formula(formula)
    features_dict = calculate_features(material_dict, elements_df)

    # XGBoost needs a DataFrame, not a dictionary
    return pd.DataFrame([features_dict])
if __name__ == "__main__":
    main()