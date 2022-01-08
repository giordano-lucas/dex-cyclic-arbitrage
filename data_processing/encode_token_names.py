# transform tokens into dummies
from sklearn.preprocessing import OneHotEncoder

def one_hot_tokens():
    one_enc = OneHotEncoder(sparse=False) 
    unique_tokens = np.unique(pd.concat([log_data[token] for token in token_columns],axis=0))
    one_enc.fit(unique_tokens.reshape(-1, 1))
    # transform
    encode = lambda col: one_enc.transform(log_data[col].to_numpy().reshape(-1, 1))
    # encode and convert as dataframes
    
    encoded = [pd.DataFrame(encode(token)).add_prefix(f"token_{key}_") for key,token in enumerate(token_columns)]
    return pd.concat(encoded, axis='columns')



token_columns = list(filter(lambda c: c.startswith('token'),log_data.columns))
X = pd.concat([log_data.drop(columns=token_columns),one_hot_tokens()], axis='columns')