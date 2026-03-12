import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = ['age', 'gender', 'zip3']
    
    anon_unique = anon_df.drop_duplicates(subset=quasi_identifiers, keep=False)
    aux_unique = aux_df.drop_duplicates(subset=quasi_identifiers, keep=False)
    
    matches_df = pd.merge(anon_unique, aux_unique, on=quasi_identifiers, how='inner')
    
    return matches_df[['anon_id', 'name']].rename(columns={'name': 'matched_name'})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0

    return len(matches_df) / len(anon_df)
