import mdtraj as md
import numpy as np
import scipy
import sys
from matplotlib import pyplot as plt
import igraph as ig
from igraph import *
import leidenalg as la
from sklearn.decomposition import PCA
import seaborn as sns
from collections import Counter
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import KernelDensity
import argparse


parser = argparse.ArgumentParser(
    description='MDlon') 

parser.add_argument('-x', '--crystal', required=True,
                    help='PDB structure in pdb format.')

parser.add_argument('-t', '--traj', required=True,
                    help='MD trajectory in xtc format.')

parser.add_argument('-e', '--energy_file', required=True,
                    help='xvg file containing energies for each frame in the trajectory.')

parser.add_argument('-s', '--stride', default = 10, required=False,
                    help='Determines the number of conformations to keep for MDlon analysis. A stride of 10 keeps 1 frame in 10. It is recommended to keep 1000 frames.')

parser.add_argument('-sg', '--savegraph', default = '', required=False,
                    help='Filename for saving the MDlon graph on disk.')

parser.add_argument('-lg', '--loadgraph', default = '', required=False,
                    help='Filename for loading the MDlon graph from disk.')

parser.add_argument('--get_structures', action = 'store_true', default = False, required=False,
                    help='Extract representative PDB structures from MD simulation data.')


parser.add_argument('-sp', '--savepartition', default = '', required=False,
                    help='Filename for saving the Leiden partition with Fruchterman Reingold layout on disk.')

parser.add_argument('-sd', '--savesampling', default = '', required=False, 
                    help='Filename for saving a dataframe with conformation sampling data in csv format on disk.')

parser.add_argument('-sh', '--savehisto', default = '', required=False,
                    help='Filename for saving a conformation sampling histogram on disk.')

parser.add_argument('-spca', '--savepca', default = '', required=False,
                    help='Filename for saving pca plot on disk.')

parser.add_argument('-stsne', '--savetsne', default = '', required=False,
                    help='Filename for saving tsne plot on disk.')

parser.add_argument('-sl', '--savelandscape', default = '', required=False,
                    help='Filename for saving 3d free energy landscape plot on disk.')

parser.add_argument('-pc', '--plotcommunities', default = 'comm_graph.pdf', required=False,
                    help='Filename for saving communities graph on disk.')

parser.add_argument('--dumpcsv', default = '', required=False,
                    help='Filename for saving csv with all data on communities.')

args = parser.parse_args()


crystal_fn = args.crystal
trajectory_fn = args.traj
energy_file = args.energy_file
stride = int(args.stride)
graph_fn = args.savegraph
load_fn = args.loadgraph
leiden_fn = args.savepartition
samplingcsv_fn = args.savesampling
samplinghisto_fn = args.savehisto
pca_fn = args.savepca
tsne_fn = args.savetsne
landscape_fn = args.savelandscape
get_structures = args.get_structures
dumpcsv_fn = args.dumpcsv
comm_file = args.plotcommunities

crystal = md.load(crystal_fn)
trajectory = md.load(trajectory_fn, stride = stride, top = crystal)


sns.set_style('whitegrid')
sns.despine(left=True, bottom=True)

def cal_rmsds(trajectory):  
    n = trajectory.n_frames
    rmsd_arr = np.empty([n,n])
    for i in range(n):
        rmsds_to_crystal = md.rmsd(trajectory, trajectory, i)
        rmsd_arr[i,:]=rmsds_to_crystal
    return rmsd_arr

def is_neighbor(col,arr):
    compare_line = arr < np.reshape(arr[:,col],(-1,1)) # For each line i of the matrice returns True for the column j if arr[i,j] < arr[i,col]
    compare_col = arr[:,col] < np.reshape(arr[:,col],(-1,1)) #  For column col, returns True for each element smaller than the element i (i go through the column)
    count = np.sum( compare_line & compare_col, axis = 1) # compare_line & compare_colreturns True if there exists a conformation closer to the conformation 'i' and to the conformation 'col' than they are between each other
    return count > 0

def relative_neighbors(arr):
    y = arr.shape[1]
    has_closer_conf = np.array([is_neighbor(col,arr) for col in range(y)]).T
    neighbors = np.logical_not(has_closer_conf)
    np.fill_diagonal(neighbors,False)
    return neighbors

def get_energies(trajectory, energy_file):
    _, ener = np.loadtxt(energy_file,comments=["@","#"],unpack= True)
    ener = [ener[i] for i in np.arange(len(ener), step=stride)]
    absmin_ener = np.abs(np.min(ener))
    ener = (ener/absmin_ener)*100 # Normalize energies so that minimum is not lower than -100
    return ener


def compute_local_min(arr_isneighbor,energy):
    local_min_dic = {}
    x,y = arr_isneighbor.shape
    for i in range(x):
        lm = True
        for j in range(y):
            if arr_isneighbor[i,j] == True:             
                if  energy[j]< energy[i]:
                    lm = False
                    break        
        if lm == True:           
            local_min_dic[i]=energy[i]
    return local_min_dic

def build_neighbors_graph(rel_neigh_arr, ener):
    g =  ig.Graph.Adjacency((rel_neigh_arr > 0).tolist(),'directed') 
    probas = []
    for e in g.es:
        probas.append(np.exp(ener[e.source]-ener[e.target])) # P(target)/P(source) = transition probability from source to target
    g.es['probas'] = probas
    for v in g.vs:
        edges_from_source = g.incident(v, mode="out")
        neigh_probas = []
        for efs in edges_from_source:
            neigh_probas.append(g.es[efs]['probas'])
            normalized_probas = neigh_probas / np.sum(neigh_probas)
        g.es[edges_from_source]['probas'] = normalized_probas
    g.es['weights'] = [-np.log(x) for x in g.es['probas']] # Switch back to energies, so that we can sum up when computing paths (equivalent to multiplying probas).
    return g

def compute_basin_node(g, i, ener):
    basin = set()
    for n in g.neighbors(i):
        if (ener[n]>ener[i]):    
            basin.update(compute_basin_node(g,n, ener)) # Recursively add neighbors with higher energy for each neighbor with higher energy to the basin
            basin.update({n})    # Add current neighbor with higher energy to the basin 
    return basin
        
def compute_basins(g, local_min, ener):  
    # Initialize a dictionary of basin for all local minima
    basins = {}
    for n in local_min:
        basin = compute_basin_node(g,n, ener)
        basins[n]=basin
    return basins

def prob_a2b(g, all_basins,a,b):
    # Check the overlapping nodes between the basins of the local optimum a and the local optimum b
    common_nodes = list(all_basins[a] & all_basins[b])
    # If there are common nodes, there exists a path from local optimum a to local optimum b
    if len(common_nodes) > 0:
        path =  g.get_shortest_paths(v = a, to = b, weights = 'weights', output = 'vpath')[0] # Shortests_paths returns a list of lists, we only have one distance to compute
        prob = 1
        for i in range(len(path)-1): # 
            prob = prob * g.es.find(_between=((path[i],),(path[i+1],)))['probas'] # Multiplying all the probabilities on the path in order to get proba from a to b
        return prob
    return 0

def community_intersection(dfopt, all_basins, i, j):
    lm_i = dfopt.index[dfopt['Community']==i+1].to_list() # Communities start at 1
    lm_j = dfopt.index[dfopt['Community']==j+1].to_list()
    return any(all_basins[x] & all_basins[y] for x in lm_i for y in lm_j)

def compute_probabilities(g, all_basins, local_min_list):
    min2min_probas = []
    for min1 in local_min_list:
        min1_out_proba = []
        for min2 in local_min_list:
            if min1 != min2:
               min1_out_proba.append(prob_a2b(g, all_basins, min1, min2))
            else: 
                min1_out_proba.append(0)
        min2min_probas.append(min1_out_proba)
    return min2min_probas

def compute_rep_probabilities(g, all_basins, communities, dfopt):
    probas = []
    point_list = list(communities['Representative'])
    partitions = list(communities['Partition'])
    for i,x in enumerate(point_list):
        x_proba = []
        for j,y in enumerate(point_list):
            if x != y and community_intersection(dfopt, all_basins, i, j):
                path = g.get_shortest_paths(v = x, to = y, weights='weights', output = 'vpath')[0]
                prob = 1
                for k in range(len(path)-1): # 
                    prob = prob * g.es.find(_between=((path[k],),(path[k+1],)))['probas'] # Multiplying all the probabilities on the path in order to get proba from a to b
                x_proba.append(prob)
            else: 
                x_proba.append(0)
        probas.append(x_proba)
    return probas

def build_lon(trajectory, rmsd_array, relneigh_g, all_basins, local_min, ener):
    print("Computing neighbors graph")
    mdlon_probas = compute_probabilities(relneigh_g, all_basins, list(local_min.keys()))
    mdlon_g =  ig.Graph.Adjacency((np.asarray(mdlon_probas) > 0).tolist(),'directed') 
    mdlon_g.es['weights'] = [i for s in mdlon_probas for i in s if i>0] # Assigns a weight to each edge which corresponds to the probability of going from one local minimum to the other
    for v in mdlon_g.vs:
        edges_from_source = mdlon_g.incident(v, mode="out")
        neigh_probas = []
        if (len(edges_from_source)>0):
            for efs in edges_from_source:
                neigh_probas.append(mdlon_g.es[efs]['weights'])
                normalized_probas = neigh_probas / np.sum(neigh_probas)
            mdlon_g.es[edges_from_source]['weights'] = normalized_probas
    mdlon_g.vs['frame'] = list(local_min.keys())
    return mdlon_g

def build_comgraph(communities, dfopt, all_basins, g):
    rep2rep_probas = compute_rep_probabilities(g, all_basins, communities, dfopt)
    rep2rep_g = ig.Graph.Adjacency((np.asarray(rep2rep_probas) > 0).tolist(),'directed') 
    rep2rep_g.es['weights'] = [i for s in rep2rep_probas for i in s if i>0]
    rep2rep_g.vs['frame'] = list(communities['Representative'])
    rep2rep_g.vs['size'] = list(communities['Size'])
    return rep2rep_g
    
    

def global_sampling(partition, lm_list, nb_frames, nb_steps = 5000): 
    visited_conf = []
    per_community_confs = [[] for _ in range(len(partition))]
    global_lon_dict = {}
    for v in mdlon_g.vs:
        edges_from_source = mdlon_g.incident(v, mode="out")
        global_lon_dict[v.index]=list(zip([mdlon_g.es[e].target for e in edges_from_source], mdlon_g.es[edges_from_source]['weights']))
    for cpt in range(100):
        start_index = np.random.randint(0,len(lm_list))
        current_conf = start_index
        per_community_confs[partition.membership[current_conf]].append(current_conf)
        visited_conf.append(current_conf)
        for i in range(nb_steps):
            indices, neigh_probas = zip(*global_lon_dict[current_conf])
            prev = current_conf
            current_conf = np.random.choice(indices, p = neigh_probas) # draw the next conformation index proportionally to its probability to be reached
            per_community_confs[partition.membership[current_conf]].append(current_conf)
            visited_conf.append(current_conf)
    global_conformation_count = Counter(visited_conf)
    df_sampling = pd.DataFrame()
    df_sampling['Frame'] = range(nb_frames)
    df_sampling['Count'] = [0]*nb_frames
    df_sampling['Partition'] = [0]*nb_frames
    for i,v in enumerate(lm_list):
        df_sampling.loc[v, 'Count'] = global_conformation_count[i]
        df_sampling.loc[v, 'Partition'] = partition.membership[i]+1
    df_sampling['Frame'] = df_sampling['Frame'].astype(int)
    df_sampling['Partition'] = df_sampling['Partition'].astype(int)
    df_sampling['Count'] = df_sampling['Count'].astype(int)
    return df_sampling, per_community_confs

def plot_sampling_histogram(fn, per_community_confs, lm_list):
    sns.set_palette("bright")

    flat_data = []
    for idx, sublist in enumerate(per_community_confs):
        for value in sublist:
            flat_data.append([lm_list[value], f'{idx+1}'])

    df = pd.DataFrame(flat_data, columns=['Frame', 'Community'])
    histo = sns.histplot(data=df, x='Frame', hue='Community', discrete=True, binwidth=1)
    sns.despine(left=True)
    ax = plt.gca()
    ax.xaxis.grid(False)

    fig =histo.get_figure()
    fig.set_figwidth(12)
    fig.savefig(fn, dpi=300)

def compute_PCA(trajectory):
    pca = PCA(n_components=2)
    trajectory.superpose(trajectory, 0)
    reduced_cartesian = pca.fit_transform(trajectory.xyz.reshape(trajectory.n_frames, trajectory.n_atoms * 3))
    return reduced_cartesian

def compute_tSNE(rmsd_array):
    tsne = TSNE(n_components=2, metric = 'precomputed', init='random').fit_transform(rmsd_array)
    return tsne

def get_representatives(df_sampling, top=-1):
    grouped = df_sampling.groupby('Partition').size().reset_index(name='Size')
    sorted_communities = grouped.sort_values(by='Size', ascending=False)
    sorted_communities = sorted_communities[sorted_communities['Partition']!=0].reset_index(drop=True)
    for community in sorted_communities['Partition']:
        community_df = df_sampling[df_sampling['Partition']==community]
        rep = community_df['Count'].idxmax()
        sorted_communities.loc[sorted_communities['Partition']==community,'Representative'] = rep
    sorted_communities['Representative'] = sorted_communities['Representative'].astype(int)
    return sorted_communities
    
def build_representation_dataframes(partition, pca, tsne, lm_list, reps):
    df = pd.DataFrame()
    df['PCA 1'] = pca[:,0]
    df['PCA 2'] = pca[:,1]
    df['TSNE 1'] = tsne[:,0]
    df['TSNE 2'] = tsne[:,1]
    df['Type'] = 'Regular'
    df['Size'] = 1
    df['Community'] = np.nan
    for i in lm_list:
        df.loc[i,'Type'] = 'Local optimum'
        df.loc[i,'Size'] = 5
    for i in reps:
        df.loc[i,'Type'] = 'Representative conformation'
        df.loc[i,'Size'] = 10
    for i in range(len(partition)):
        for j in partition[i]:
            df.loc[lm_list[j], 'Community'] = i+1
    dfreg = df[df['Type']=='Regular']
    dfopt = df[df['Type']!='Regular']
    dfrep = df[df['Type']=='Representative conformation']
    return df, dfreg, dfopt, dfrep

def plot_pca(df, dfreg, pca_fn):
    plt.figure()
    sns.scatterplot(data = dfreg, x = 'PCA 1', y='PCA 2', color = 'lightgray', markers='.', style='Type', legend=False)
    pcaplot = sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Community', palette='colorblind', style='Type', markers=['.','o', 'D'])
    sns.scatterplot(data=df, x='PCA 1', y='PCA 2', hue='Community', palette='colorblind',  size='Size', style='Type', markers=['.','o','D'], legend=False, ax=pcaplot)
    handles, labels = pcaplot.get_legend_handles_labels()
    int_labels = []
    for label in labels:
        try:
            int_label = int(float(label))
            int_labels.append(str(int_label))
        except ValueError:
            int_labels.append(label)

    pcaplot.legend(handles, int_labels)

    sns.move_legend(pcaplot, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(pca_fn, dpi=300, bbox_inches='tight')

def plot_tsne(df, dfreg, tsne_fn):
    plt.figure()
   
        
    sns.scatterplot(data = dfreg, x = 'TSNE 1', y='TSNE 2', color = 'lightgray', markers='.', style='Type', legend=False)
    tsneplot = sns.scatterplot(data=df, x='TSNE 1', y='TSNE 2', hue='Community', palette='colorblind', style='Type', markers=['.','o','D'])
    sns.scatterplot(data=df, x='TSNE 1', y='TSNE 2', hue='Community', palette='colorblind',  size='Size', style='Type', markers=['.','o','D'], legend=False, ax=tsneplot)
    handles, labels = tsneplot.get_legend_handles_labels()

    int_labels = []
    for label in labels:
        try:
            int_label = int(float(label))
            int_labels.append(str(int_label))
        except ValueError:
            int_labels.append(label)

    tsneplot.legend(handles, int_labels)

    sns.move_legend(tsneplot, "upper left", bbox_to_anchor=(1, 1))
    plt.savefig(tsne_fn, dpi=300, bbox_inches='tight')

def plot_3dlandscape(tsne, df_sampling, lm_list, landscape_fn):
    kde = KernelDensity(kernel='gaussian', bandwidth=7)
    sx = [tsne[i,0] for i in lm_list]
    sy = [tsne[i,1] for i in lm_list]
    data = np.vstack((sx,sy)).T
    kdeweights = [df_sampling.loc[i,'Count'] for i in lm_list]
    kde.fit(data, sample_weight=kdeweights)
    radius = np.max(np.linalg.norm(data, axis=1))+10
    num_points = 500
    theta, r = np.meshgrid(np.linspace(0, 2*np.pi, num_points), np.linspace(0, radius, num_points))
    X = r * np.cos(theta)
    Y = r * np.sin(theta)
    positions = np.vstack([X.ravel(), Y.ravel()]).T
    log_density = kde.score_samples(positions)
    density = np.exp(log_density)
    zmax = max(density)
    density = density.reshape(X.shape)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, -density+zmax, cmap='coolwarm_r', linewidth=0, antialiased=True)
    ax.scatter(X,Y, c=-density, cmap='coolwarm_r', s=2, alpha=0.1)
    plt.axis('off')
    ax.view_init(30,0)
    plt.savefig(landscape_fn, dpi=300)

def extract_frames(trajectory, reps):
    for i,v in enumerate(reps):
        pdb_name = "community_{}".format(i+1)
        pdb_file = "{}.pdb".format(pdb_name)
        trajectory[v].save(pdb_file)

def plot_communities(fn, partition):
    aggregated_partition = partition.aggregate_partition() # one node per community
    comm_graph = aggregated_partition.graph.simplify()
    edge_list = comm_graph.get_edgelist()

    all_weights = {}
    for v in comm_graph.vs:
        all_weights[v.index] = 0
        
    for edge in edge_list:
        s,e = edge
        weight = aggregated_partition.weight_to_comm(s,e)
        all_weights[s]+=weight
        comm_graph.es[comm_graph.get_eid(s,e)]['edgewidth'] = weight

    maxweight = max(comm_graph.es['edgewidth'])
    minweight = min(comm_graph.es['edgewidth'])
    minscale = 0.2
    maxscale = 1
    for e in comm_graph.es:
        if (maxweight-minweight != 0):
            e['edgecolor'] = minscale + (e['edgewidth']-minweight) * ((maxscale-minscale)/(maxweight - minweight))
        else:
            e['edgecolor'] = 0.5
        
    for v in comm_graph.vs:
        v['reached'] = aggregated_partition.total_weight_to_comm(v.index) #all_weights[v.index] # switch to all_weights in order to exclude intra-community weights
    maxreached = max(comm_graph.vs['reached'])
    comm_graph.vs['reached'] = [0 if v < 0 else v/maxreached for v in comm_graph.vs['reached']]
    comm_graph.vs['labels'] = ['Community {}'.format(i) for i in range(1, len(aggregated_partition)+1)]
    maxweight = max(comm_graph.es['edgewidth'])
    comm_graph.es['edgewidth'] = [w/maxweight for w in comm_graph.es['edgewidth']]
    minwidth = min(comm_graph.es['edgewidth'])
    color_gradient = plt.get_cmap('coolwarm')
    color_gradient_edges = plt.get_cmap('Greys')
    
    node_colors = [color_gradient(value) for value in comm_graph.vs['reached']]
    edge_colors = [color_gradient_edges(value) for value in comm_graph.es['edgecolor']]
    bbox = BoundingBox(600, 600)
    ig.plot(comm_graph, fn, vertex_size = partition.sizes(), vertex_label = comm_graph.vs['labels'], vertex_label_dist = 2, vertex_color = node_colors, edge_color = edge_colors, edge_width=1, edge_arrow_size = 0.5, layout='kk', dpi=300, bbox = bbox, margin=(80,80,80,80))


print("Computing RMSD")
rmsd_array = cal_rmsds(trajectory)
print("Extracting energies from MD data")
ener = get_energies(trajectory, energy_file)
print("Building relative neighborhood graph")
rel_neigh_arr = relative_neighbors(rmsd_array)
relneigh_g = build_neighbors_graph(rel_neigh_arr, ener)
print("Finding local minima")
local_min = compute_local_min(rel_neigh_arr, ener)
print("Computing all basins of attraction")
all_basins = compute_basins(relneigh_g, local_min, ener)


if load_fn:
    print("loading LON from disk")
    mdlon_g = ig.Graph.Read_GraphML(load_fn)
else:
    print("Building LON")
    mdlon_g = build_lon(trajectory, rmsd_array, relneigh_g, all_basins, local_min, ener)

if graph_fn:
    print("Saving graph")
    mdlon_g.write(graph_fn)

print("Finding communities")
partition = la.find_partition(mdlon_g, weights = 'weights', partition_type=la.ModularityVertexPartition, n_iterations=1000)
print("Quality of partitioning : "+str(partition.quality()))

plot_communities(comm_file, partition)

if leiden_fn:
    print("Plotting leiden partition")
    ig.plot(partition, leiden_fn, edge_arrow_size = 0.4, layout='fruchterman_reingold')

lm_list = [int(i) for i in mdlon_g.vs['frame']]

print("Sampling LON")
df_sampling, per_comm_confs = global_sampling(partition, lm_list, trajectory.n_frames)

if samplingcsv_fn:
    print("Saving sampling data to csv file")
    df_sampling.to_csv(samplingcsv_fn)

if samplinghisto_fn:
    print("Plotting sampling histogram")
    plot_sampling_histogram(samplinghisto_fn, per_comm_confs, lm_list)

print("Computing PCA")
pca = compute_PCA(trajectory)
print("Computing t-SNE")
tsne = compute_tSNE(rmsd_array)

community_df = get_representatives(df_sampling)
print("Communities : ")
print(community_df)
reps = list(community_df['Representative'])

if (get_structures):
    print("Extracting representative frames")
    extract_frames(trajectory, reps)
    
df, dfreg, dfopt, dfrep = build_representation_dataframes(partition, pca, tsne, lm_list, reps)

print("Building community graph")
comm_g = build_comgraph(community_df, dfopt, all_basins, relneigh_g)



if dumpcsv_fn:
    print("Saving all data to csv file")
    df.to_csv(dumpcsv_fn)


if pca_fn:
    print("Plotting PCA")
    plot_pca(df, dfreg, pca_fn)

if tsne_fn:
    print("Plotting t-SNE")
    plot_tsne(df, dfreg, tsne_fn)

if landscape_fn:
    print("Plotting 3D free energy landscape")
    plot_3dlandscape(tsne, df_sampling, lm_list, landscape_fn)




