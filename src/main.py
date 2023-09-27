import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags
import os
import csv
import time


def GPR(transition_matrix, alpha, d, max_iter=1000, tol=1e-8):
    n = transition_matrix.shape[0]
    p0 = np.ones(n) / n
    p = p0.copy()
    for i in range(max_iter):
        p_old = p.copy()
        p = (1 - alpha) * (transition_matrix.T @ p_old + d @ p_old) + alpha * p0
        if np.linalg.norm(p - p_old, ord=1) < tol:
            break

    return p


transition_file = "./data/transition.txt"
alpha = 0.2

# read transition data from file
rows, cols, _ = np.loadtxt(transition_file, dtype=int, unpack=True)
max_index = max(max(rows), max(cols))
rows -= 1  # convert to 0-based index
cols -= 1  # convert to 0-based index
values = np.ones_like(rows)  # all entries are 1

# create transition matrix
transition_mat = csc_matrix((values, (rows, cols)), shape=(max_index, max_index), dtype=float)

# normalize rows and handle dangling nodes
row_sums = transition_mat.sum(axis=1).A1
dangling = np.where(row_sums == 0)[0]
row_sums[row_sums == 0] = 1
diagonal = 1 / row_sums
transition_mat = diags(diagonal).dot(transition_mat)

d = np.zeros(max_index)
d[dangling] = 1 / max_index

# compute pagerank
GPR_start = time.time()
pagerank_vector = GPR(transition_mat, alpha, d)
GPR_time = time.time() - GPR_start
print ("GPR Pagerank time is: ", GPR_time)


# output result
with open("../GPR.txt", "w") as output_file:
    for i, score in enumerate(pagerank_vector):
        output_file.write(f"{i+1} {score}\n")


def consolidate_indri_lists():
    """
    Consolidate all indri-list files into a single text file.
    """
    # Get a list of all indri-list files
    indri_list_dir = './data/indri-lists'
    files = [f for f in os.listdir(indri_list_dir) if f.endswith('.txt')]
    
    # Read each file and consolidate the lines
    consolidated_lines = []
    for file in files:
        with open(os.path.join(indri_list_dir, file), 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                # Use the file name (without extension) as the query ID,
                # and strip '.results' from the query ID
                query_id = os.path.splitext(file)[0]
                query_id = query_id.replace('.results', '')
                row[0] = query_id
                consolidated_lines.append(row)
    
    # Write the consolidated lines to a new file
    output_file = './data/Wholetext.txt'
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(consolidated_lines)
    
    print("Finish consolidating indri-list files into one text file")

# Usage
consolidate_indri_lists()


def create_tree_eval_file_for_gpr(text_dir, IPV, method, alpha=0.5, beta=0.5):
    '''
    Create TreeEval file for GPR algorithm.
    Input: text_dir(Wholetext.txt), IPV(r vector), method(0: NS, 1:WS, 2:CM)
    '''

    # Read data into a pandas dataframe
    df = pd.read_csv(text_dir, sep=' ', header=None, names=['UID_Q', 'Q0', 'Index', 'Rank', 'Score', 'Run'])
    df['Index'] = df['Index'] - 1  # adjust index
    df[['UID', 'Q']] = df['UID_Q'].str.split('-', expand=True)
    df['Q'] = df['Q'].astype(int) - 1

    # Apply the selected method to calculate the new score
    if method == 0:
        df['NewScore'] = IPV[df['Index'], 0]
        file_name = './GPR-NS.txt'
    elif method == 1:
        df['NewScore'] = 0.5 * IPV[df['Index'], 0] + 0.5 * df['Score']
        file_name = './GPR-WS.txt'
    else:
        df['NewScore'] = alpha * np.log(IPV[df['Index'], 0]) + beta * df['Score']
        file_name = './GPR-CM.txt'
    
    # Sort by UID_Q and NewScore
    df = df.sort_values(by=['UID_Q', 'NewScore'], ascending=[True, False])
    df['NewRank'] = df.groupby('UID_Q').cumcount() + 1

    # Write to file
    with open(file_name, 'w') as f:
        for index, row in df.iterrows():
            f.write('{} Q0 {} {} {} run-1\n'.format(row['UID_Q'], row['Index']+1, row['NewRank'], row['NewScore']))
    
    print(f'Finish making the {file_name}')

text_dir = './data/Wholetext.txt'
IPV = np.loadtxt('../GPR.txt')[:, 1:2]    # r vector
alpha = 0.1  # weight for log(IPV)
beta = 0.9 # weight for original score

for method in range(3):  # 0: NS, 1: WS, 2: CM
    Retrieval_start = time.time()
    create_tree_eval_file_for_gpr(text_dir, IPV, method, alpha, beta)
    Retrieval_time = time.time() - Retrieval_start
    print ("GPR retrieval time for", method, "is: ", Retrieval_time)



#TSPR
rows, cols, _ = np.loadtxt(transition_file, dtype=int, unpack=True)
max_index = max(max(rows), max(cols))

doc_topics_file = "./data/doc_topics.txt"

_, topics = np.loadtxt(doc_topics_file, dtype=int, unpack=True)
num_topics = len(np.unique(topics))

topic_docs = [[] for _ in range(num_topics)]

with open(doc_topics_file, 'r') as f:
    for line in f:
        docid, topicid = map(int, line.split())
        topic_docs[topicid - 1].append(docid - 1)

# create pt for each topic
pt = np.zeros((num_topics, max_index))

for topicid in range(num_topics):
    pt[topicid, topic_docs[topicid]] = 1 / len(topic_docs[topicid])


beta = 0.8  
alpha = 0.1
gamma = 0.1
p0 = np.ones(max_index) / max_index

num_iterations = 100  # number of iterations for the PageRank algorithm
r_t = np.zeros((num_topics, max_index))

TSPR_start = time.time()
for t in range(num_topics):
    r_t[t] = np.ones(max_index) / max_index  # initial value of r_t
    
    for _ in range(num_iterations):
        r_t[t] = alpha * transition_mat.T.dot(r_t[t]) + beta * pt[t] + gamma * p0

TSRP_time = time.time() - TSPR_start
print ("TSRP time is: ", TSRP_time)


query_topic_distro = {}

with open('./data/query-topic-distro.txt') as f:
    for line in f:
        parts = line.split()
        user_id = int(parts[0])
        query_id = int(parts[1])
        for i in range(2, len(parts)):
            topic_id, ptq = parts[i].split(':')
            topic_id = int(topic_id)
            ptq = float(ptq)
            query_topic_distro[(user_id, query_id, topic_id)] = ptq


user_topic_distro = {}

with open('./data/user-topic-distro.txt') as f:
    for line in f:
        parts = line.split()
        user_id = int(parts[0])
        query_id = int(parts[1])
        for i in range(2, len(parts)):
            topic_id, ptq = parts[i].split(':')
            topic_id = int(topic_id)
            ptq = float(ptq)
            user_topic_distro[(user_id, query_id, topic_id)] = ptq


def get_tspr(user_id, query_id, topic_distro, rt):
    T = rt.shape[0]  # number of topics

    rq_tspr = np.zeros_like(rt[0])
    
    for t in range(T):
        ptq = topic_distro.get((user_id, query_id, t), 0)
        rq_tspr += ptq * rt[t]
    
    return rq_tspr


rq_qspr = get_tspr(2, 1, query_topic_distro, r_t)
with open("../QTSPR-U2Q1.txt", "w") as output_file:
    for i, score in enumerate(rq_qspr):
        output_file.write(f"{i+1} {score}\n")

rq_tspr = get_tspr(2, 1, user_topic_distro, r_t)
with open("../PTSPR-U2Q1.txt", "w") as output_file:
    for i, score in enumerate(rq_tspr):
        output_file.write(f"{i+1} {score}\n")



QTSPR_start = time.time()
for (user_id, query_id, _), ptq in query_topic_distro.items():
    all_rq_qtspr = get_tspr(user_id, query_id, query_topic_distro, r_t)

QTSPR_time = time.time() - QTSPR_start + TSRP_time
print("QTSPR time is: ", QTSPR_time)

PTSPR_start = time.time()
for (user_id, query_id, _), ptq in user_topic_distro.items():
    all_rq_ptspr = get_tspr(user_id, query_id, user_topic_distro, r_t)

PTSPR_time = time.time() - PTSPR_start + TSRP_time
print("PTSPR time is: ", PTSPR_time)


def create_tree_eval_file_for_tspr(text_dir, IPV, method, alpha=0.5, beta=0.5):
    '''
    Create TreeEval file for TSPR (QTSPR or PTSPR) algorithm.
    Input: text_dir(Wholetext.txt), IPV(r vector), method(0: NS, 1: WS, 2: CM), alpha, beta
    '''

    # Read data into a pandas dataframe
    df = pd.read_csv(text_dir, sep=' ', header=None, names=['UID_Q', 'Q0', 'Index', 'Rank', 'Score', 'Run'])
    df['Index'] = df['Index'] - 1  # adjust index
    df[['UID', 'Q']] = df['UID_Q'].str.split('-', expand=True)
    df['Q'] = df['Q'].astype(int) - 1

    # Apply the selected method to calculate the new score
    if np.array_equal(IPV, IPV_qtspr):
        if method == 0:
            df['NewScore'] = IPV[df['Index']]
            file_name = f'./QTSPR_NS.txt'
        elif method == 1:
            df['NewScore'] = 0.5 * IPV[df['Index']] + 0.5 * df['Score']
            file_name = f'./QTSPR_WS.txt'
        else:
            df['NewScore'] = alpha * np.log(IPV[df['Index']]) + beta * df['Score']
            file_name = f'./QTSPR_CM.txt'
    else:
        if method == 0:
            df['NewScore'] = IPV[df['Index']]
            file_name = f'./PTSPR_NS.txt'
        elif method == 1:
            df['NewScore'] = 0.5 * IPV[df['Index']] + 0.5 * df['Score']
            file_name = f'./PTSPR_WS.txt'
        else:
            df['NewScore'] = alpha * np.log(IPV[df['Index']]) + beta * df['Score']
            file_name = f'./PTSPR_CM.txt'

    
    # Sort by UID_Q and NewScore
    df = df.sort_values(by=['UID_Q', 'NewScore'], ascending=[True, False])
    df['NewRank'] = df.groupby('UID_Q').cumcount() + 1

    # Write to file
    with open(file_name, 'w') as f:
        for index, row in df.iterrows():
            f.write('{} Q0 {} {} {} run-1\n'.format(row['UID_Q'], row['Index']+1, row['NewRank'], row['NewScore']))
    
    print(f'Finish making the {file_name}')

text_dir = './data/Wholetext.txt'
IPV_qtspr = all_rq_qtspr
IPV_ptspr = all_rq_ptspr 
alpha = 0.1  # weight for log(IPV)
beta = 0.9  # weight for original score


for method in [0,1,2]:
    q_retrieval_start = time.time()
    create_tree_eval_file_for_tspr(text_dir, IPV_qtspr, method, alpha, beta)
    q_retrieval_time = time.time() - q_retrieval_start
    print ("QTSPR retrieval time for", method, "is: ", q_retrieval_time)

    p_retrieval_start = time.time()
    create_tree_eval_file_for_tspr(text_dir, IPV_ptspr, method, alpha, beta)
    p_retrieval_time = time.time() - p_retrieval_start
    print ("PTSPR retrieval time for", method, "is: ", p_retrieval_time)


