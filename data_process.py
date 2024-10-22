import numpy as np
import networkx as nx
import scipy.sparse as sp
import pickle
from scipy.sparse import csr_array
from scipy.sparse import coo_array


def process_raw_snap_graph(input_file, csr_file, id_file):
    # Create an undirected graph
    nodes = dict()
    graph = nx.Graph()
    with open(input_file, 'r') as file:
        for line in file:
            # Skip comment lines and blank lines
            if line.startswith("#") or line.strip() == "":
                continue
            # Split the line into source and target nodes
            source, target = map(int, line.strip().split("\t"))
            # ignore self-loops
            if source == target:
                continue
            nodes[source] = ''
            nodes[target] = ''
            graph.add_edge(source, target)
    print(f'processing: {input_file}')
    print(f'num-nodes: {len(nodes)} num-edges: {nx.number_of_edges(graph)} '
          f'min-node-id: {min(nodes.keys())} max-node-id: {max(nodes.keys())}')
    num_cc = nx.number_connected_components(graph)
    largest_cc = []
    for cc in nx.connected_components(graph):
        largest_cc = list(cc)
        break
    print(f'num-cc: {num_cc} largest-cc: {len(largest_cc)} num-nodes: {len(nodes)}')
    unique_ids = sorted(largest_cc)
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    sub_graph = nx.Graph()
    for (u, v) in list(graph.subgraph(largest_cc).edges):
        sub_graph.add_edge(id_mapping[u], id_mapping[v])
    print(f'sub-num-nodes: {nx.number_of_nodes(sub_graph)} sub-num-edges: {nx.number_of_edges(sub_graph)} '
          f'min-node-id: {min(sub_graph.nodes)} max-node-id: {max(sub_graph.nodes)}')
    csr_mat = nx.to_scipy_sparse_array(
        sub_graph, nodelist=range(nx.number_of_nodes(sub_graph)),
        dtype=np.float32, weight=None, format='csr')
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(unique_ids))


def process_raw_ogbn_graph(edge_index, csr_file, id_file):
    # Create an undirected graph
    nodes = dict()
    graph = nx.Graph()
    row = edge_index[0, :]  # Source nodes
    col = edge_index[1, :]  # Target nodes
    for source, target in zip(row, col):
        nodes[source] = ''
        nodes[target] = ''
        if source == target:
            print('self-loops', source, target)
            continue
        graph.add_edge(source, target)
    print(max(nodes.keys()), min(nodes.keys()), len(nodes))
    n = nx.number_of_nodes(graph)
    m = nx.number_of_edges(graph)
    num_cc = nx.number_connected_components(graph)
    largest_cc = []
    for cc in nx.connected_components(graph):
        largest_cc = list(cc)
        break
    print(n, m, num_cc, len(largest_cc))
    unique_ids = sorted(largest_cc)
    print(unique_ids[0], unique_ids[-1])
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    sub_graph = nx.Graph()
    for (u, v) in list(graph.subgraph(largest_cc).edges):
        sub_graph.add_edge(id_mapping[u], id_mapping[v])
    print(max(sub_graph.nodes), min(sub_graph.nodes), nx.number_connected_components(sub_graph))
    csr_mat = nx.to_scipy_sparse_array(
        sub_graph, nodelist=range(nx.number_of_nodes(sub_graph)),
        dtype=np.float64, weight=None, format='csr')
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(unique_ids))


def process_com_dblp(input_file="dataset/com-dblp/com-dblp.ungraph.txt",
                     csr_file=f"dataset/com-dblp/com-dblp_csr-mat.npz",
                     id_file=f"dataset/com-dblp/com-dblp_id-mapping.npz"):
    not_largest_cc = 0
    edge_index = 0
    for i in range(len(edge_index[0])):
        uu, vv = edge_index[0, i], edge_index[1, i]
        if not_largest_cc[uu] is True or not_largest_cc[vv] is True:
            pass
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_as_skitter(input_file="dataset/as-skitter/as-skitter.txt",
                       csr_file=f"dataset/as-skitter/as-skitter_csr-mat.npz",
                       id_file=f"dataset/as-skitter/as-skitter_id-mapping.npz"):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_com_lj(input_file="dataset/com-lj/com-lj.ungraph.txt",
                   csr_file=f"dataset/com-lj/com-lj_csr-mat.npz",
                   id_file=f"dataset/com-lj/com-lj_id-mapping.npz"):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_com_orkut(input_file="dataset/com-orkut/com-orkut.ungraph.txt",
                      csr_file=f"dataset/com-orkut/com-orkut_csr-mat.npz",
                      id_file=f"dataset/com-orkut/com-orkut_id-mapping.npz"):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_wiki_talk(input_file=f"dataset/wiki-talk/wiki-Talk.txt",
                      csr_file=f"dataset/wiki-talk/wiki-talk_csr-mat.npz",
                      id_file='dataset/wiki-talk/wiki-talk_id-mapping.npz'):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_soc_lj1(input_file=f"dataset/soc-lj1/soc-LiveJournal1.txt",
                    csr_file=f"dataset/soc-lj1/soc-lj1_csr-mat.npz",
                    id_file='dataset/soc-lj1/soc-lj1_id-mapping.npz'):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_cit_patents(input_file=f"dataset/cit-patent/cit-Patents.txt",
                        csr_file=f"dataset/cit-patent/cit-patent_csr-mat.npz",
                        id_file='dataset/cit-patent/cit-patent_id-mapping.npz'):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_soc_pokec(input_file=f"dataset/soc-pokec/soc-pokec-relationships.txt",
                      csr_file=f"dataset/soc-pokec/soc-pokec_csr-mat.npz",
                      id_file='dataset/soc-pokec/soc-pokec_id-mapping.npz'):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_com_youtube(input_file="dataset/com-youtube/com-youtube.ungraph.txt",
                        csr_file=f"dataset/com-youtube/com-youtube_csr-mat.npz",
                        id_file=f"dataset/com-youtube/com-youtube_id-mapping.npz"):
    process_raw_snap_graph(input_file, csr_file, id_file)


def process_com_friendster_mapping(input_file="dataset/com-friendster/com-friendster.ungraph.txt",
                                   id_file=f"dataset/com-friendster/com-friendster_id-mapping.npz",
                                   mapped_file=f"dataset/com-friendster/com-friendster.ungraph-mapped.txt"):
    nodes = dict()
    self_loops = 0.
    ind = 0
    with open(input_file, 'r') as file:
        for line in file:
            # Skip comment lines and blank lines
            if line.startswith("#") or line.strip() == "":
                continue
            # Split the line into source and target nodes
            source, target = map(int, line.strip().split("\t"))
            if source == target:
                print('self-loop', source, target)
                self_loops += 1.
            nodes[source] = ''
            nodes[target] = ''
            if ind % 1000000 == 0:
                print(f'{ind / 1000000}M')
            ind += 1
    print(max(nodes.keys()), min(nodes.keys()), len(nodes))
    unique_ids = sorted(list(nodes.keys()))
    np.savez(id_file, np.asarray(unique_ids))
    unique_ids = np.load(id_file)['arr_0']
    print(unique_ids[0], unique_ids[-1])
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    with open(mapped_file, 'w') as file_w:
        with open(input_file, 'r') as file_r:
            for line in file_r:
                # Skip comment lines and blank lines
                if line.startswith("#") or line.strip() == "":
                    continue
                # Split the line into source and target nodes
                source, target = map(int, line.strip().split("\t"))
                file_w.write(f"{id_mapping[source]} {id_mapping[target]}\n")
    file_w.close()


def process_com_friendster(input_file="dataset/com-friendster/com-friendster.ungraph-mapped.txt",
                           csr_file=f"dataset/com-friendster/com-friendster_csr-mat.npz"):
    ind = 0
    num_nodes = 65608366
    num_edges = 1806067135
    edges = np.zeros(shape=(2, 2 * num_edges), dtype=np.int32)
    with open(input_file, 'r') as file:
        for line in file:
            source, target = map(int, line.strip().split(" "))
            edges[0, ind] = source
            edges[1, ind] = target
            edges[0, ind + num_edges] = target
            edges[1, ind + num_edges] = source
            ind += 1
            if ind % 1000000 == 0:
                print(f'{ind / 1000000}M')
    assert ind == num_edges
    ones = np.ones(2 * num_edges, np.float32)
    from scipy.sparse import csr_array
    csr_mat = csr_array(
        (ones, (edges[0, :], edges[1, :])), shape=(num_nodes, num_nodes))
    sp.save_npz(csr_file, csr_mat)


def process_wiki_en21(input_file="dataset/wiki-en21/wiki-en21-edges-undirected.txt",
                      csr_file=f"dataset/wiki-en21/wiki-en21_csr-mat.npz",
                      id_file=f"dataset/wiki-en21/wiki-en21_id-mapping.npz"):
    nodes = dict()
    self_loops = 0.
    ind = 0
    edges = []
    with open(input_file, 'r') as file:
        for line in file:
            # Skip comment lines and blank lines
            if line.startswith("#") or line.strip() == "":
                continue
            # Split the line into source and target nodes
            source, target = map(int, line.strip().split(" "))
            if source == target:
                print('self-loop', source, target)
                self_loops += 1.
                continue
            nodes[source] = ''
            nodes[target] = ''
            if ind % 1000000 == 0:
                print(f'{ind / 1000000}M')
            ind += 1
            edges.append((source, target))
    print(max(nodes.keys()), min(nodes.keys()), len(nodes))
    unique_ids = sorted(list(nodes.keys()))
    print(unique_ids[0], unique_ids[-1], self_loops)
    # checked it is a single connected component
    graph = nx.from_edgelist(edgelist=edges, create_using=nx.Graph)
    print(nx.number_of_edges(graph), len(edges))
    csr_mat = nx.to_scipy_sparse_array(
        graph, nodelist=range(nx.number_of_nodes(graph)),
        dtype=np.float32, weight=None, format='csr')
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(unique_ids))


def process_ogbn_arxiv(d_name='ogbn-arxiv'):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name=d_name)
    # graph: library-agnostic graph object
    graph, label = dataset[0]
    edge_index = graph['edge_index']
    process_raw_ogbn_graph(edge_index,
                           csr_file=f"dataset/ogbn-arxiv/ogbn-arxiv_csr-mat.npz",
                           id_file=f"dataset/ogbn-arxiv/ogbn-arxiv_id-mapping.npz")


def process_ogbn_products(d_name='ogbn-products'):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name=d_name)
    # graph: library-agnostic graph object
    graph, label = dataset[0]
    print('load_ogbn_products')
    edge_index = graph['edge_index']
    process_raw_ogbn_graph(edge_index,
                           csr_file=f"dataset/ogbn-products/ogbn-products_csr-mat.npz",
                           id_file=f"dataset/ogbn-products/ogbn-products_id-mapping.npz")


def process_ogbn_mag(d_name='ogbn-mag'):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name=d_name)
    # graph: library-agnostic graph object
    graph, label = dataset[0]
    print('load_ogbn_mag')
    g1 = graph['edge_index_dict'][('author', 'affiliated_with', 'institution')]
    g2 = graph['edge_index_dict'][('author', 'writes', 'paper')]
    g3 = graph['edge_index_dict'][('paper', 'cites', 'paper')]
    g4 = graph['edge_index_dict'][('paper', 'has_topic', 'field_of_study')]

    nodes = dict()
    edges = []
    index_node = 0
    for u, v in zip(g1[0, :], g1[1, :]):
        uu, vv = 'author' + str(u), 'institution' + str(v)
        if uu not in nodes:
            nodes[uu] = index_node
            index_node += 1
        if vv not in nodes:
            nodes[vv] = index_node
            index_node += 1
        edges.append([nodes[uu], nodes[vv]])
    for u, v in zip(g2[0, :], g2[1, :]):
        uu, vv = 'author' + str(u), 'paper' + str(v)
        if uu not in nodes:
            nodes[uu] = index_node
            index_node += 1
        if vv not in nodes:
            nodes[vv] = index_node
            index_node += 1
        edges.append([nodes[uu], nodes[vv]])
    for u, v in zip(g3[0, :], g3[1, :]):
        uu, vv = 'paper' + str(u), 'paper' + str(v)
        if uu not in nodes:
            nodes[uu] = index_node
            index_node += 1
        if vv not in nodes:
            nodes[vv] = index_node
            index_node += 1
        edges.append([nodes[uu], nodes[vv]])
    for u, v in zip(g4[0, :], g4[1, :]):
        uu, vv = 'paper' + str(u), 'field_of_study' + str(v)
        if uu not in nodes:
            nodes[uu] = index_node
            index_node += 1
        if vv not in nodes:
            nodes[vv] = index_node
            index_node += 1
        edges.append([nodes[uu], nodes[vv]])
    edge_index = np.zeros(shape=(2, len(edges)))
    edge_index[0, :] = [_[0] for _ in edges]
    edge_index[1, :] = [_[1] for _ in edges]
    print(len(edge_index), len(nodes))
    process_raw_ogbn_graph(edge_index,
                           csr_file=f"dataset/ogbn-mag/ogbn-mag_csr-mat.npz",
                           id_file=f"dataset/ogbn-mag/ogbn-mag_id-mapping.npz")


def process_ogbn_proteins(d_name='ogbn-proteins'):
    from ogb.nodeproppred import NodePropPredDataset
    dataset = NodePropPredDataset(name=d_name)
    # graph: library-agnostic graph object
    graph, label = dataset[0]
    print('load_ogbn_proteins')
    edge_index = graph['edge_index']
    process_raw_ogbn_graph(edge_index,
                           csr_file=f"dataset/ogbn-proteins/ogbn-proteins_csr-mat.npz",
                           id_file=f"dataset/ogbn-proteins/ogbn-proteins_id-mapping.npz")


def process_ogbn_papers_100M_preprocessing():
    print('Loading necessary files...')
    print('This might take a while.')
    data_dict = np.load('/mnt/data1/jiahe/ApprCode/dataset/ogbn_papers100M/raw/data.npz')
    print(data_dict.__dict__)
    # node_feat, edge_index, num_nodes_list, num_edges_list, node_year
    edge_index = data_dict['edge_index']
    id_file = f"/mnt/data1/jiahe/ApprCode/dataset/ogbn-papers100M/ogb-papers100M_edge-index.npz"
    np.savez(id_file, edge_index)
    print("success!")


def process_ogbn_papers_100M_preprocess2():
    edge_index_file = f"/mnt/data1/jiahe/ApprCode/dataset/ogbn-papers100M/ogb-papers100M_edge-index.npz"
    edge_index = np.load(edge_index_file)['arr_0']
    csr_file = f"dataset/ogbn-papers100M/ogbn-papers100M_csr-mat.npz"
    id_file = f"dataset/ogbn-papers100M/ogbn-papers100M_id-mapping.npz"
    num_nodes = 111059956
    num_edges = 1615685872
    print(num_nodes, num_edges)
    # convert coo_matrix to undirected to csr
    ind = 0
    edges = np.zeros(shape=(2, 2 * num_edges), dtype=np.int32)
    for ind in range(len(edge_index[0, :])):
        source, target = edge_index[0, ind], edge_index[1, ind]
        edges[0, ind] = source
        edges[1, ind] = target
        edges[0, ind + num_edges] = target
        edges[1, ind + num_edges] = source
        ind += 1
        if ind % 1000000 == 0:
            print(f'{ind / 1000000}M')
    assert ind == num_edges
    ones = np.ones(2 * num_edges, np.float32)
    from scipy.sparse import csr_array
    csr_mat = csr_array(
        (ones, (edges[0, :], edges[1, :])), shape=(num_nodes, num_nodes))
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.arange(111059956))

    from scipy.sparse.csgraph import breadth_first_order
    list_nodes, _ = breadth_first_order(csr_mat, 0)
    print(len(list_nodes), 111059956)
    sp.save_npz(csr_file, csr_mat)


def process_ogbn_papers_100M_preprocess3():
    edge_index_file = f"{root_path()}/dataset/ogbn-papers100M/ogb-papers100M_edge-index.npz"
    edge_index = np.load(edge_index_file)['arr_0']
    csr_file = f"dataset/ogbn-papers100M/ogbn-papers100M_csr-mat.npz"
    id_file = f"dataset/ogbn-papers100M/ogbn-papers100M_id-mapping.npz"
    num_nodes = 111059956
    num_edges = 1615685872
    print(num_nodes, num_edges)
    print(len(edge_index[0]), np.min(edge_index[0]), np.max(edge_index[0]))
    print(len(edge_index[1]), np.min(edge_index[1]), np.max(edge_index[1]))
    largest_cc = pickle.load(open(f"dataset/ogbn-papers100M/largest-cc.pkl", 'rb'))
    unique_ids = sorted(largest_cc)
    print(unique_ids[0], unique_ids[-1])
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    not_largest_cc = np.ones(num_nodes, dtype=np.bool_)
    not_largest_cc[unique_ids] = False
    print('not-largest-cc', np.nonzero(not_largest_cc))
    print(len(id_mapping))
    out_edges = []
    for i in range(len(edge_index[0])):
        uu, vv = edge_index[0, i], edge_index[1, i]
        if not_largest_cc[uu] == 1 or not_largest_cc[vv] == 1:
            out_edges.append(i)
        else:
            edge_index[0, i] = id_mapping[edge_index[0, i]]
            edge_index[1, i] = id_mapping[edge_index[1, i]]
    # save not largest cc edges to file
    print('not-largest-cc-edges:', out_edges)
    pickle.dump(out_edges, open(f"dataset/ogbn-papers100M/not-in-largest-cc-edges.pkl", 'wb'))
    num_sub_edges = num_edges - len(out_edges)
    num_sub_nodes = len(unique_ids)
    xx = np.ones(num_edges)
    xx[out_edges] = 0
    sub_edge_index = np.nonzero(xx)[0]
    edges = np.zeros(shape=(2, 2 * num_sub_edges), dtype=np.int32)
    edges[0, :len(sub_edge_index)] = edge_index[0, sub_edge_index]
    edges[0, len(sub_edge_index):] = edge_index[1, sub_edge_index]
    edges[1, :len(sub_edge_index)] = edge_index[1, sub_edge_index]
    edges[1, len(sub_edge_index):] = edge_index[0, sub_edge_index]
    ones = np.ones(2 * num_sub_edges, np.float32)
    csr_mat = csr_array((ones, (edges[0, :], edges[1, :])),
                        shape=(num_sub_nodes, num_sub_nodes))
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(unique_ids))


def process_ogbn_papers_100M():
    edge_index_file = f"{root_path()}/dataset/ogbn-papers100M/ogb-papers100M_edge-index.npz"
    edge_index = np.load(edge_index_file)['arr_0']
    csr_file = f"dataset/ogbn-papers100M/ogbn-papers100M_csr-mat.npz"
    id_file = f"dataset/ogbn-papers100M/ogbn-papers100M_id-mapping.npz"
    num_nodes = 111059956
    num_edges = 1615685872
    print(num_nodes, num_edges)
    print(len(edge_index[0]), np.min(edge_index[0]), np.max(edge_index[0]))
    print(len(edge_index[1]), np.min(edge_index[1]), np.max(edge_index[1]))
    largest_cc = pickle.load(open(f"dataset/ogbn-papers100M/largest-cc.pkl", 'rb'))
    unique_ids = sorted(largest_cc)
    print(unique_ids[0], unique_ids[-1])
    not_largest_cc = np.ones(num_nodes, dtype=np.bool_)
    not_largest_cc[unique_ids] = False
    out_edges = pickle.load(
        open(f"dataset/ogbn-papers100M/not-in-largest-cc-edges.pkl", 'rb'))
    out_edges = []
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
    for i in range(len(edge_index[0])):
        uu, vv = edge_index[0, i], edge_index[1, i]
        if not_largest_cc[uu] == True or not_largest_cc[vv] == True:
            out_edges.append(i)
        else:
            edge_index[0, i] = id_mapping[edge_index[0, i]]
            edge_index[1, i] = id_mapping[edge_index[1, i]]

    num_sub_edges = num_edges - len(out_edges)
    num_sub_nodes = len(unique_ids)
    xx = np.ones(num_edges)
    xx[out_edges] = 0
    sub_edge_index = np.nonzero(xx)[0]
    np.savez("dataset/ogbn-papers100M/"
             "ogbn-papers100M_mapped-edge-index-0.npz",
             row=edge_index[0, sub_edge_index])
    np.savez("dataset/ogbn-papers100M/"
             "ogbn-papers100M_mapped-edge-index-1.npz",
             col=edge_index[1, sub_edge_index])
    edges = np.zeros(shape=(2, 2 * num_sub_edges), dtype=np.int32)
    edges[0, :len(sub_edge_index)] = edge_index[0, sub_edge_index]
    edges[0, len(sub_edge_index):] = edge_index[1, sub_edge_index]
    edges[1, :len(sub_edge_index)] = edge_index[1, sub_edge_index]
    edges[1, len(sub_edge_index):] = edge_index[0, sub_edge_index]
    ones = np.ones(2 * num_sub_edges, dtype=np.float32)
    csr_mat = csr_array((ones, (edges[0, :], edges[1, :])),
                        shape=(num_sub_nodes, num_sub_nodes))
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(unique_ids))


def process_ogbl_ppa(d_name='ogbl-ppa'):
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name=d_name)[0]
    # graph: library-agnostic graph object
    edge_index = dataset['edge_index']
    process_raw_ogbn_graph(edge_index,
                           csr_file=f"dataset/ogbl-ppa/ogbl-ppa_csr-mat.npz",
                           id_file=f"dataset/ogbl-ppa/ogbl-ppa_id-mapping.npz")


def process_lsc_mag240m_preprocessing():
    from ogb.lsc import MAG240MDataset
    dataset = MAG240MDataset(root="dataset/")
    print(dataset.__dict__)
    print(dataset.num_papers)  # number of paper nodes
    print(dataset.num_authors)  # number of author nodes
    print(dataset.num_institutions)  # number of institution nodes
    print(dataset.num_paper_features)  # dimensionality of paper features
    print(dataset.num_classes)  # number of subject area classes
    edge_index_writes = dataset.edge_index('author', 'paper')
    id_file = f"dataset/ogb-mag240m/ogb-mag240m_edge-index-writes.npz"
    np.savez(id_file, edge_index_writes)
    edge_index_cites = dataset.edge_index('paper', 'paper')
    id_file = f"dataset/ogb-mag240m/ogb-mag240m_edge-index-cites.npz"
    np.savez(id_file, edge_index_cites)
    edge_index_affiliated_with = dataset.edge_index('author', 'institution')
    id_file = f"dataset/ogb-mag240m/ogb-mag240m_edge-index-affiliated-with.npz"
    np.savez(id_file, edge_index_affiliated_with)


def process_lsc_mag240m():
    root = '/mnt/data1/jiahe/ApprCode/dataset/mag240m_kddcup2021/mag240m_kddcup2021/processed/'

    # n = 244,160,499
    # m =   1,728,364,232
    # institutions: 25,721
    # authors: 122,383,112
    # papers: 121,751,666
    # 3,456,728,464
    # 244,160,500

    id_file = root + f"author___affiliated_with___institution/edge_index.npy"
    edge_index1 = np.load(id_file)
    num_edges1 = len(edge_index1[0, :])
    edge_index1[0, :] += 25721
    print(len(edge_index1[0, :]))
    print(np.min(edge_index1[0, :]), np.max(edge_index1[0, :]))
    print(np.min(edge_index1[1, :]), np.max(edge_index1[1, :]))
    # 44,592,586
    # 2 122,383,104 authors
    # 0 25,720 institutions
    # 38,682,409

    id_file = root + f"author___writes___paper/edge_index.npy"
    edge_index2 = np.load(id_file)
    num_edges2 = len(edge_index2[0, :])
    edge_index2[0, :] += 25721
    edge_index2[1, :] += 122383112 + 25721
    print(len(edge_index2[0, :]))
    print(np.min(edge_index2[0, :]), np.max(edge_index2[0, :]))
    print(np.min(edge_index2[1, :]), np.max(edge_index2[1, :]))
    # 386,022,720
    # 0 122,383,111 authors
    # 0 121,751,665 institutions
    # 122,383,112

    id_file = root + f"paper___cites___paper/edge_index.npy"
    edge_index3 = np.load(id_file)
    edge_index3[0, :] += 122383112 + 25721
    edge_index3[1, :] += 122383112 + 25721
    num_edges3 = len(edge_index3[0, :])
    print(len(edge_index3[0, :]))
    print(np.min(edge_index3[0, :]), np.max(edge_index3[0, :]))
    print(np.min(edge_index3[1, :]), np.max(edge_index3[1, :]))
    # 1,297,748,926
    # 3 121,751,664
    # 1 121,751,664
    # 72,508,661
    # 244,160,499
    num_edges = num_edges1 + num_edges2 + num_edges3
    # 122383112
    num_nodes = 244160499
    csr_file = f"dataset/ogb-mag240m/ogb-mag240m_csr-mat.npz"
    id_file = f"dataset/ogb-mag240m/ogb-mag240m_id-mapping.npz"
    csr_mat = coo_array((
        np.ones(2 * num_edges, dtype=np.float32),
        (np.concatenate([edge_index1[0, :], edge_index2[0, :], edge_index3[0, :],
                         edge_index1[1, :], edge_index2[1, :], edge_index3[1, :]]),
         np.concatenate([edge_index1[1, :], edge_index2[1, :], edge_index3[1, :],
                         edge_index1[0, :], edge_index2[0, :], edge_index3[0, :]]))),
        shape=(num_nodes, num_nodes)).tocsr()
    sp.save_npz(csr_file, csr_mat)
    np.savez(id_file, np.asarray(np.arange(num_nodes)))


def process_middle_scale():
    process_as_skitter()
    process_cit_patents()
    process_com_dblp()
    process_com_lj()
    process_com_orkut()
    process_com_youtube()
    process_soc_lj1()
    process_soc_pokec()
    process_wiki_talk()


def process_middle_scale_ogbn():
    process_ogbn_products()
    process_ogbn_arxiv()
    process_ogbn_mag()
    process_ogbn_proteins()


if __name__ == '__main__':
    # process_com_lj()
    # very large one
    # process_ogbn_papers_100M()
    process_lsc_mag240m()
