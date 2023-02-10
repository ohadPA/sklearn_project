import pandas

def create_output_df(data):
    output_df = []
    IDs = range(2001, 4001)
    list_of_tuples = list(zip(IDs, data))
    list_of_tuples
    output_df = pandas.DataFrame(list_of_tuples, columns=['ID', 'rating'])
    return output_df


def create_output_df(data):
    output_df = []
    IDs = range(30001, 50001)
    list_of_tuples = list(zip(IDs, data))
    list_of_tuples
    output_df = pandas.DataFrame(list_of_tuples, columns=['ID', 'rating'])
    return output_df

def create_output_csv(filename, df):
    df.to_csv(filename, sep=',', index=False)

def bag_all_positive_words(row, columns, is_rollout):
    review_text = ""
    for col in columns:
        if row[col] > 0 and col!="ID" and col!="rating":
            review_text+= col +" "
    return review_text

def bag_all_negative_words(row, columns, is_rollout):
    review_text = ""
    for col in columns:
        if row[col] == 1 and col!="ID" and col!="rating":
            review_text+= col +" "
    return review_text

def add_review_column(df, is_rollout):
    output_df = df
    output_df['review'] = df.apply(lambda row: bag_all_positive_words(row, df.columns, is_rollout), axis=1)
    # output_df['pos_review'] = df.apply(lambda row: bag_all_positive_words(row, df.columns, is_rollout), axis=1)
    # output_df['neg_review'] = df.apply(lambda row: bag_all_negative_words(row, df.columns, is_rollout), axis=1)
    return output_df

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==1 and y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==0 and y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return[TP, FP, TN, FN]