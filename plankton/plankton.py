
from __future__ import annotations
from cgitb import text
from enum import unique
from hashlib import new
# from msilib import add_data
from turtle import color

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

from typing import Union

from matplotlib import pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.cm import get_cmap
from matplotlib.widgets import Button, TextBox
import matplotlib.patheffects as PathEffects
import pickle

from scipy import sparse
import collections
import scanpy as sc
import anndata
import scipy
import random

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

from umap import UMAP
from sklearn.manifold import TSNE

plt.style.use('dark_background')

class PixelMap():

    def __init__(self: PixelMap,
                 pixel_data: np.ndarray,
                 upscale: float = 1.0,) -> None:

        self.data = pixel_data

        self.n_channels = 1 if len(
            pixel_data.shape) == 2 else pixel_data.shape[-1]

        if not isinstance(upscale, collections.Iterable) or len(upscale) == 1:
            self.scale = (upscale, upscale)
        else:
            self.scale = upscale

        self.extent = (0, pixel_data.shape[0] / self.scale[0], 0,
                       pixel_data.shape[1] / self.scale[1])

    @property
    def shape(self):
        return self.extent[1] - self.extent[0], self.extent[3] - self.extent[2]

    def imshow(self, axd=None, **kwargs) -> None:
        extent = np.array(self.extent)

        if (len(self.data.shape)>2) and (self.data.shape[2]>4):
            data = self.data.sum(-1)
        else:
            data = self.data

        if axd is None:
            axd = plt.subplot(111)

        axd.imshow(data, extent=extent[[0, 3, 1, 2]], **kwargs)

    def __getitem__(self, indices: Union[slice, collections.Iterable[slice]]):

        if not isinstance(indices, collections.Iterable):
            index_x = indices
            index_y = slice(0, None, None)

        else:
            index_x = indices[0]

            if len(indices) > 1:
                index_y = indices[1]
            else:
                index_y = slice(0, None, None)

        if (index_x.start is None):
            start_x = 0  # self.extent[0]
        else:
            start_x = index_x.start
        if (index_x.stop is None):
            stop_x = self.extent[1]
        else:
            stop_x = index_x.stop

        if (index_y.start is None):
            start_y = 0  # self.extent[2]
        else:
            start_y = index_y.start
        if (index_y.stop is None):
            stop_y = self.extent[3]
        else:
            stop_y = index_y.stop

        data = self.data[int(start_y * self.scale[1]):int(stop_y *
                                                          self.scale[1]),
                         int(start_x * self.scale[0]):int(stop_x *
                                                          self.scale[0]), ]

        return PixelMap(
            data,
            upscale=self.scale,
        )

class KDEProjection(PixelMap):
    def __init__(self,sd: SpatialData,
                 bandwidth: float = 3.0,
                 threshold_vf_norm: float = 1.0,
                 threshold_p_corr: float = 0.5,
                 upscale: float = 1) -> None:
        
        self.sd = sd
        self.bandwidth = bandwidth
        self.threshold_vf_norm = threshold_vf_norm
        self.threshold_p_corr = threshold_p_corr

        self.scale = upscale

        super().__init__(self.run_kde(), upscale)



    def run_kde(self) -> None:

        kernel = self.generate_kernel(self.bandwidth*3, self.scale)

        x_int = np.array(self.sd.y * self.scale).astype(int)
        y_int = np.array(self.sd.x * self.scale).astype(int)
        genes = self.sd.gene_ids

        vf = np.zeros((x_int.max()+kernel.shape[0]+1,y_int.max()+kernel.shape[0]+1,len(self.sd.genes)))

        for x,y,g in zip(x_int,y_int,genes):
            # print(x,y,vf.shape,kernel.shape)
            vf[x:x+kernel.shape[0],y:y+kernel.shape[1],g]+=kernel
            
        return vf[kernel.shape[0]//2:-kernel.shape[0]//2,kernel.shape[1]//2:-kernel.shape[1]//2]

    def generate_kernel(self, bandwidth: float, scale: float = 1) -> np.ndarray:

        kernel_width_in_pixels = int(bandwidth * scale *
                                     6)  # kernel is 3 sigmas wide.

        span = np.linspace(-3, 3, kernel_width_in_pixels)
        X, Y = np.meshgrid(span, span)

        return 1 / (2 * np.pi)**0.5 * np.exp(-0.5 * ((X**2 + Y**2)**0.5)**2)


class CellTypeMap(PixelMap):
    def __init__(self,data,celltype_labels,*args,**kwargs):
        
        # .super().

        pass
    # def __init__(self: PixelMap, pixel_data: np.ndarray, upscale: float = 1) -> None:
    #     super().__init__(pixel_data, upscale)



class SpatialGraph():

    def __init__(self, df, n_neighbors=10) -> None:

        self.df = df
        self.n_neighbors = n_neighbors
        self._neighbors = None
        self._neighbor_types = None
        self._distances = None
        self._umap = None
        self._tsne = None

    @property
    def neighbors(self):
        if self._neighbors is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbors[:,:self.n_neighbors]

    @property
    def distances(self):
        if self._distances is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._distances[:,:self.n_neighbors]

    @property
    def neighbor_types(self):
        if self._neighbor_types is None:
            self._distances, self._neighbors, self._neighbor_types = self.update_knn(
                self.n_neighbors)
        return self._neighbor_types[:,:self.n_neighbors]

    @property
    def umap(self):
        if self._umap is None:
            self.run_umap()
        return self._umap

    @property
    def tsne(self):
        if self._tsne is None:
            self.run_tsne()
        return self._tsne

    def __getitem__(self,*args):
        sg = SpatialGraph(self.df,self.n_neighbors)
        if self._distances is not None:
            sg._distances = self._distances.__getitem__(*args)
        if self._neighbors is not None:
            sg._neighbors = self._neighbors.__getitem__(*args)
        if self._neighbor_types is not None:
            sg._neighbor_types = self._neighbor_types.__getitem__(*args)

    def update_knn(self, n_neighbors, re_run=False):

        if self._neighbors is not None and (n_neighbors <
                                            self._neighbors.shape[1]):
            self.n_neighbors=n_neighbors
            # return (self._neighbors[:, :n_neighbors],
            #         self._distances[:, :n_neighbors],
            #         self._neighbor_types[:, :n_neighbors])
        else:

            coordinates = np.stack([self.df.x, self.df.y]).T
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(coordinates)
            self._distances, self._neighbors = knn.kneighbors(coordinates)
            self._neighbor_types = np.array(self.df.gene_ids)[self._neighbors]

            self.n_neighbors = n_neighbors

            # return self.distances, self.neighbors, self.neighbor_types

    def knn_entropy(self, n_neighbors=4):

        self.update_knn(n_neighbors=n_neighbors)
        indices = self.neighbors  # (n_neighbors=n_neighbors)

        knn_cells = np.zeros_like(indices)
        for i in range(indices.shape[1]):
            knn_cells[:, i] = self.df['gene_id'].iloc[indices[:, i]]

        H = np.zeros((len(self.df.genes), ))

        for i, gene in enumerate(self.df.genes):
            x = knn_cells[self.df['gene_id'] == i]
            _, n_x = np.unique(x[:, 1:], return_counts=True)
            p_x = n_x / n_x.sum()
            h_x = -(p_x * np.log2(p_x)).sum()
            H[i] = h_x

        return (H)

    def plot_entropy(self, n_neighbors=4):

        H = self.knn_entropy(n_neighbors)

        idcs = np.argsort(H)
        plt.figure(figsize=(25, 25))

        fig, axd = plt.subplot_mosaic([
            ['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
            ['bar', 'bar', 'bar', 'bar'],
            ['scatter_5', 'scatter_6', 'scatter_7', 'scatter_8'],
        ],
            figsize=(11, 7),
            constrained_layout=True)

        dem_plots = np.array([
            0,
            2,
            len(H) - 3,
            len(H) - 1,
            1,
            int(len(H) / 2),
            int(len(H) / 2) + 1,
            len(H) - 2,
        ])
        colors = ('royalblue', 'goldenrod', 'red', 'purple', 'lime',
                  'turquoise', 'green', 'yellow')

        axd['bar'].bar(
            range(len(H)),
            H[idcs],
            color=[
                colors[np.where(
                    dem_plots == i)[0][0]] if i in dem_plots else 'grey'
                for i in range(len(self.df.stats.counts))
            ])

        axd['bar'].set_xticks(range(len(H)),
                              [self.df.genes[h] for h in idcs],
                              rotation=90)
        axd['bar'].set_ylabel('knn entropy, k=' + str(n_neighbors))

        for i in range(8):
            idx = idcs[dem_plots[i]]
            gene = self.df.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.df.x, self.df.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.df.x[self.df['gene_id'] == idx],
                                   self.df.y[self.df['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')
            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            if i < 4:
                y_ = (H[idcs])[i]
                _y = 0
            else:
                y_ = 0
                _y = 1

            con = ConnectionPatch(xyA=(dem_plots[i], y_),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[_y]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

    def _determine_counts(self,bandwidth=1, kernel=None):

        counts = np.zeros((len(self.df,),len(self.df.genes)))
        if kernel is None:
            kernel = lambda x: np.exp(-x**2/(2*bandwidth**2))

        for i in range(0,self.n_neighbors):
            counts[np.arange(len(self.df)),self.neighbor_types[:,i]]+=  kernel(self.distances[:,i])
        return counts

    def run_umap(self,bandwidth=1,kernel=None,metric='cosine', zero_weight=1,*args,**kwargs):        
        # print(kwargs)
        counts = self._determine_counts(bandwidth=bandwidth,kernel=kernel)
        assert (all(counts.sum(1))>0)
        counts[np.arange(len(self.df)),self.df.gene_ids]+=zero_weight-1
        umap=UMAP(metric=metric,*args,**kwargs)
        self._umap = umap.fit_transform(counts)

    # def run_tsne(self,bandwidth=1,kernel=None,*args,**kwargs):        
    #     counts = self._determine_counts(bandwidth=bandwidth,kernel=kernel)
    #     tsne=TSNE(*args,**kwargs)
    #     self._tsne = tsne.fit_transform(counts)

    # def plot_umap(self, text_prop=None, color_prop='genes', color_dict=None, c=None,text_distance=1, thlds_text=(1.0,0.0,None),text_kwargs={}, **kwargs):
    #     self.plot_embedding(self.umap, text_prop=text_prop, color_prop=color_prop, color_dict=color_dict, 
    #     c=c,text_distance=text_distance, thlds_text=thlds_text, text_kwargs=text_kwargs, **kwargs)

    # def plot_tsne(self,text_prop=None, color_prop='genes', color_dict=None, c=None,text_distance=1, **kwargs):
    #     self.plot_embedding(self.tsne, text_prop=text_prop, color_prop=color_prop, color_dict=color_dict, c=c,text_distance=text_distance, **kwargs)

    def plot_umap(self, color_prop='genes', text_prop=None, 
                    text_color_prop = None, c=None, color=None, color_dict=None,text_distance=1, thlds_text=(1.0,0.0,None), text_kwargs={}, **kwargs):

        embedding=self.umap

        categories = self.df.props[color_prop].unique() 
        colors = self.df.props.project('c_'+color_prop)
        handlers = [plt.scatter([],[],color=self.df.props[self.df.props[color_prop]==c]['c_'+color_prop][0]) for c in categories]

        if color is not None:
            colors=(color,)*len(self.df)
        if c is not None:
            colors=c

        plt.legend(handlers, categories)

        plt.scatter(*embedding.T,c=colors, **kwargs)



        if text_prop is not None:

            text_xs=[]
            text_ys=[]
            text_cs=[]
            text_is=[]
            text_zs=[]

            X, Y = np.mgrid[embedding[:,0].min():embedding[:,0].max():100j, 
                embedding[:,1].min():embedding[:,1].max():100j]
            positions = np.vstack([X.ravel(), Y.ravel()])

           
            for i,g in enumerate(self.df.props[text_prop].unique()):
                if text_color_prop is not None:
                    fontcolor = self.df.props[self.df.props[text_prop]==g]['c_'+text_color_prop][0]
                else: fontcolor='w'

                # try:
                embedding_subset = embedding[self.df.g.isin(self.df.props[self.df.props[text_prop]==g].index)].T
                if embedding_subset.shape[1]>2:
                    kernel = scipy.stats.gaussian_kde(embedding_subset)

                    Z = np.reshape(kernel(positions).T, X.shape)
                    localmaxs = scipy.ndimage.maximum_filter(Z,size=(10,10))
                    maxz = (localmaxs==Z)&(localmaxs>=(localmaxs.max()*thlds_text[0]))
                    maxs = np.where(maxz)
                    # maxz = Z.max()
                    maxx = X[:,0][maxs[0]]
                    maxy = Y[0][maxs[1]]
                
                    for j in range(len(maxx)):
                        plt.suptitle(maxx[j])
                        text_xs.append(maxx[j])
                        text_ys.append(maxy[j])
                        text_is.append(g)
                        text_cs.append(fontcolor)
                        text_zs.append(Z[maxs[0][j],maxs[1][j]])

            cogs = self._untangle_text(np.array([text_xs,text_ys,]).T, min_distance=text_distance)
            for i,c in enumerate(cogs):
                # plt.suptitle(self.df.counts[text_is[i]])
                if (text_zs[i]>(max(text_zs)*thlds_text[1])):
                    if (thlds_text[2] is not None) and (text_is[i] in self.df.counts ) and (self.df.counts[text_is[i]]>thlds_text[2]):
                        txt = plt.text(c[0],c[1],text_is[i],color=text_cs[i],ha='center',**text_kwargs)
                        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

                # except:
                #     print(f'Failed to assign text label to {g}')

    def umap_interactive(self, color_prop='genes', text_prop='genes'):


        click_coords=np.array((0.0,0.0))

        plt.style.use('dark_background')

        fig = plt.gcf()

        radius=5
        # colors = np.repeat([(0,1,0,1)],len(sdata),axis=0)

        ax1 = plt.subplot2grid((3, 2), (0, 0),2,1)
        
        sc2,_,_ = self.df.scatter(marker='x',alpha=0.5,s=[10]*len(self.df),axd=ax1)

        ax2 = plt.subplot2grid((3, 2), (0, 1),2,1)  
        self.df.graph.plot_umap(color_prop=color_prop, text_prop=text_prop, alpha=0.1, text_distance=0.5,marker='x',thlds_text=(0.8,0.2,0.1),text_kwargs={'fontsize':10})
        plt.legend([],[])

        circle=plt.Circle((0,0),radius, color='w',fill=False, linestyle='--')
        ax2.add_artist(circle)

        ax3 = plt.subplot2grid((3, 2), (2, 0),1,1)

        bars=ax3.bar(range(10),np.zeros((10,)))
        plt.xticks(range(10),['-']*10,rotation=90)
        ax3.set_ylim(0,1)
        ax3.set_ylabel('molecule counts')

        # ax4 = plt.subplot2grid((3, 2), (2, 1),1,1)
        ax4=plt.axes([0.6,0.28,0.2,0.05])
        ax5=plt.axes([0.6,0.2,0.2,0.05])
        ax6=plt.axes([0.8,0.2,0.1,0.05])
        ax7=plt.axes([0.6,0.12,0.2,0.05])

        dist=np.zeros((len(self.df),))

        text_box_radius = TextBox(ax4, 'radius', initial=radius,color='lightgray',hovercolor='darkgray')
        text_box_radius.text_disp.set_color('k')

        text_box_label = TextBox(ax5, 'label', initial='mask_1',color='lightgray',hovercolor='darkgray')
        text_box_label.text_disp.set_color('k')

        txt = ax7.text(0.0,0.5,'',color='r',va='center')
        ax7.set_axis_off()

        def store_selection(event):
            # print('kewl',dist)
            txt.set_text(text_box_label.text)

            if (~(text_box_label.text in self.df.columns)) or ('Click again to overwrite.' in  txt.text):
                
                txt.set_text((dist<int(text_box_radius.text)).sum())
                self.df[text_box_label.text]=dist<int(text_box_radius.text)
                txt.set_text(f'Stored {(dist<int(text_box_radius.text)).sum()} points as {text_box_label.text}.')
            
            elif (text_box_label.text in self.df.columns):
                # txt.set_text('kewlest')
                txt.set_text(f'Label {text_box_label.text} already exists. Click again to overwrite.')
                

            fig.canvas.draw()

        btn_submit=Button(ax6,'store selection',color='rosybrown',hovercolor='lightcoral')
        btn_submit.on_clicked(store_selection)

        def on_click(event):
            click_coords[0]=event.xdata
            click_coords[1]=event.ydata

        def update_bars(subset):

            ax2.set_title('T')
            subset=self.df[subset]
            heights = subset.stats.counts.sort_values()[-10:]
            ax3.set_ylim(0,heights.max())

            [b.set_height(heights.iloc[i]) for i,b in enumerate(bars)]
            ax3.set_xticklabels(heights.index)      
            [b.set_color(subset.props['c_'+color_prop][heights.index[i]]) for i,b in enumerate(bars)]        
            
        def on_release(event):
            center = np.array((event.xdata,event.ydata))
            if all(click_coords==center) and (event.inaxes is ax2):
                # ax2.set_title(f'{event.xdata},{event.ydata}')
                circle.set_center(center)
                
                dist[:] = ((self.df.graph.umap-center)**2).sum(1)**0.5
                
                radius=int(text_box_radius.text)
                text_box_label.text
                btn_submit
                circle.radius=radius
                
                sc2._sizes[dist<radius]=20
                sc2._sizes[dist>radius]=0

                update_bars(dist<radius)
                        
                fig.canvas.draw()
                
                
                
        plt.connect('button_press_event', on_click)
        plt.connect('button_release_event', on_release)
            


 
    def _untangle_text(self, cogs, untangle_rounds=50, min_distance=0.5):
        knn = NearestNeighbors(n_neighbors=2)

        np.random.seed(42)
        cogs_new = cogs+np.random.normal(size=cogs.shape,)*0.01

        for i in range(untangle_rounds):

            cogs = cogs_new.copy()
            knn = NearestNeighbors(n_neighbors=2)

            knn.fit(cogs)
            distances,neighbors = knn.kneighbors(cogs/[1.01,1.0])
            too_close = (distances[:,1]<min_distance)

            for i,c in enumerate(np.where(too_close)[0]):
                partner = neighbors[c,1]
                cog = cogs[c]-cogs[partner]
                cog_new = cogs[c]+0.3*cog
                cogs_new[c]= cog_new
                
        return cogs_new


class ScanpyDataFrame():

    def __init__(self, sd, scanpy_ds):
        self.sd = sd
        self.adata = scanpy_ds
        self.stats = ScStatistics(self)
        self.celltype_labels = None
        self.signature_matrix = None

    @property
    def shape(self):
        return self.adata.shape

    def generate_signatures(self, celltype_obs_marker='celltype'):

        self.celltype_labels = np.unique(self.adata.obs[celltype_obs_marker])

        self.signature_matrix = np.zeros((
            len(self.celltype_labels),
            self.adata.shape[1],
        ))

        for i, label in enumerate(self.celltype_labels):
            self.signature_matrix[i] = np.array(
                self.adata[self.adata.obs[celltype_obs_marker] == label].X.sum(
                    0)).flatten()

        self.signature_matrix = self.signature_matrix - self.signature_matrix.mean(
            1)[:, None]
        self.signature_matrix = self.signature_matrix / self.signature_matrix.std(
            1)[:, None]

        self.signature_matrix = pd.DataFrame(self.signature_matrix, index=self.celltype_labels,columns=self.stats.index )

        return self.signature_matrix

    def synchronize(self):

        joined_genes = (self.stats.genes.intersection(self.sd.genes)).sort_values()

        # print(len(joined_genes))

        self.sd.reset_index()
        self.adata = self.adata[:, joined_genes]
        self.stats = ScStatistics(self)
        self.sd.drop(index=list(
            self.sd.index[~self.sd.g.isin(joined_genes)]),
            inplace=True)

        self.sd.stats = PointStatistics(self.sd)

        self.sd.graph = SpatialGraph(self.sd)


    def determine_gains(self):

        sc_genes = self.adata.var.index
        counts_sc = np.array(self.adata.X.sum(0) / self.adata.X.sum()).flatten()
        
        counts_spatial = np.array([self.sd.stats.get_count(g) for g in sc_genes])

        counts_spatial = counts_spatial / counts_spatial.sum()
        count_ratios = counts_sc / counts_spatial
        return count_ratios


    def plot_gains(self):

        sc_genes = self.adata.var.index

        count_ratios = self.determine_gains()
        idcs = np.argsort(count_ratios)

        ax = plt.subplot(111)

        span = np.linspace(0, 1, len(idcs))
        clrs = np.stack([
            span,
            span * 0,
            span[::-1],
        ]).T

        ax.barh(range(len(count_ratios)), np.log(count_ratios[idcs]), color=clrs)

        ax.text(0,
                len(idcs) + 3,
                'lost in spatial ->',
                ha='center',
                fontsize=12,
                color='red')
        ax.text(0, -3, '<- lost in SC', ha='center', fontsize=12, color='lime')

        for i, gene in enumerate(sc_genes[idcs]):
            if count_ratios[idcs[i]] < 1:
                ha = 'left'
                xpos = 0.05
            else:
                ha = 'right'
                xpos = -0.05
            ax.text(0, i, gene, ha=ha)

        ax.set_yticks([], [])


    def compare_counts(self):

        sc_genes = (self.adata.var.index)
        sc_counts = (np.array(self.adata.X.sum(0)).flatten())
        sc_count_idcs = np.argsort(sc_counts)
        count_ratios = np.log(self.determine_gains())
        count_ratios -= count_ratios.min()
        count_ratios /= count_ratios.max()

        ax1 = plt.subplot(311)
        ax1.set_title('compared molecule counts:')
        ax1.bar(np.arange(len(sc_counts)), sc_counts[sc_count_idcs], color='grey')
        ax1.set_ylabel('log(count) scRNAseq')
        ax1.set_xticks(np.arange(len(sc_genes)),
                    sc_genes[sc_count_idcs],
                    rotation=90)
        ax1.set_yscale('log')

        ax2 = plt.subplot(312)
        for i, gene in enumerate(sc_genes[sc_count_idcs]):
            plt.plot(
                [i, self.sd.stats.get_count_rank(gene)],
                [1, 0],
            )
        plt.axis('off')
        ax2.set_ylabel(' ')

        ax3 = plt.subplot(313)
        self.sd.plot_bars(ax3, color='grey')
        ax3.invert_yaxis()
        ax3.xaxis.tick_top()
        ax3.xaxis.set_label_position('top')
        ax3.set_ylabel('log(count) spatial')

    def score_affinity(self,labels_1,labels_2=None,scanpy_obs_label='celltype'):

        if labels_2 is None:
            labels_2 = (self.adata.obs[~self.adata.obs[scanpy_obs_label].isin(labels_1)])[scanpy_obs_label]
        
        mask_1 = self.adata.obs[scanpy_obs_label].isin(labels_1)
        mask_2 = self.adata.obs[scanpy_obs_label].isin(labels_2)
    
        samples_1 = self.adata[mask_1,]
        samples_2 = self.adata[mask_2,]

        counts_1 = np.array(samples_1.X.mean(0)).flatten()
        counts_2 = np.array(samples_2.X.mean(0)).flatten()

        return np.log((counts_1+0.1)/(counts_2+0.1))
        
                                
                                
class GeneStatistics(pd.DataFrame):
    def __init__(self, *args,**kwargs):
        super(GeneStatistics, self).__init__(*args,**kwargs)

    @property
    def counts_sorted(self):
        return self.data.counts[self.stats.count_indices]

    @property
    def genes(self):
        return self.index

    def get_count(self, gene):
        if gene in self.genes.values:
            return int(self.counts[self.genes == gene])

    def get_id(self, gene_name):
        return int(self.gene_ids[self.genes == gene_name])

    def get_count_rank(self, gene):
        if gene in self.genes.values:
            return int(self.count_ranks[self.genes == gene])

class PointStatistics(GeneStatistics):
    def __init__(self, sd):
        genes, indicers, inverse, counts = np.unique(
            sd['g'],
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(PointStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)

        sd['gene_id'] = inverse

        sd.graph = SpatialGraph(self)

class ScStatistics(GeneStatistics):

    def __init__(self, scanpy_df):

        counts = np.array(scanpy_df.adata.X.sum(0)).squeeze()
        genes = scanpy_df.adata.var.index

        count_idcs = np.argsort(counts)
        count_ranks = np.argsort(count_idcs)

        super(ScStatistics, self).__init__(
            {
                'counts': counts,
                'count_ranks': count_ranks,
                'count_indices': count_idcs,
                'gene_ids': np.arange(len(genes))
            },
            index=genes)
  
class SpatialIndexer():

    def __init__(self, df):
        self.df = df

    @property
    def shape(self):
        if self.df.background is None:
            return np.ceil(self.df.x.max() - self.df.x.min()).astype(
                int), np.ceil(self.df.y.max() - self.df.y.min()).astype(int)
        else:
            return self.df.background.shape

    def create_cropping_mask(self, start, stop, series):

        if start is None:
            start = 0

        if stop is None:
            stop = series.max()

        return ((series > start) & (series < stop))

    def join_cropping_mask(self, xlims, ylims):
        return self.create_cropping_mask(
            *xlims, self.df.x) & self.create_cropping_mask(*ylims, self.df.y)

    def crop(self, xlims, ylims):

        mask = self.join_cropping_mask(xlims, ylims)

        pixel_maps = []

        if xlims[0] is None:
            start_x = 0
        else:
            start_x = xlims[0]
        if ylims[0] is None:
            start_y = 0
        else:
            start_y = ylims[0]

        for pm in self.df.pixel_maps:
            pixel_maps.append(pm[xlims[0]:xlims[1], ylims[0]:ylims[1]])

        if self.df.scanpy is not None:
            adata = self.df.scanpy.adata
        else:
            adata=None

        return SpatialData(self.df.g[mask],
                           self.df.x[mask] - start_x,
                           self.df.y[mask] - start_y, pixel_maps,
                           adata, self.df.synchronize)

    def __getitem__(self, indices):

        if not isinstance(indices, collections.Iterable):
            indices = (indices, )
        if len(indices) == 1:
            ylims = (0, None)
        else:
            ylims = (indices[1].start, indices[1].stop)

        xlims = (indices[0].start, indices[0].stop)

        return self.crop(xlims, ylims)

class PropsDF(pd.DataFrame):

    def __init__(self, sd, assign_colors=True):
        super(PropsDF,self).__init__(index=sd.stats.index)
        self['genes']=self.index
        if assign_colors:
            self.assign_colors()
        self.sd=sd

    def assign_colors(self,label='genes',cmap=None, shuffle=False):

        # if label is None:
        #     uniques = self.index
        # else:
        uniques = self[label].unique()

        if cmap is None:
            cmap = get_cmap('nipy_spectral')    
            clrs = [cmap(f) for f in np.linspace(0.07,1,len(uniques))]
        
        else:
            cmap = get_cmap(cmap)    
            clrs = [cmap(f) for f in np.linspace(0,1,len(uniques))]
        
        if shuffle:
            random.shuffle(clrs)

        clrs = {u:clrs[i] for i,u in enumerate(uniques)}
        self['c_'+label] = self[label].apply(lambda x: clrs[x])
        # print(len(uniques),len(clrs),self.shape)
        # if label is None:
        #     self['c_genes']=clrs
        # else:
        # clr_list=[(0,0,0,0,)]*len(self)
        # for i,u in enumerate(uniques):
            # print([clrs[i]]*sum(self[label]==u))
            
            # self.loc[self[label]==u,'c_'+label]=[[clrs[i]]]*sum(self[label]==u)
        # self[]
        

    def project(self,label):
        return(self.loc[self.sd.g][label])

    def copy(self):
        return PropsDF(self.sd)

class SpatialData(pd.DataFrame):

    def __init__(self,  
                 genes,
                 x_coordinates,
                 y_coordinates,
                 pixel_maps=[],
                 scanpy=None,
                 synchronize=True,
                 props=None):

        # Initiate 'own' spot data:
        super(SpatialData, self).__init__({
            'g': genes,
            'x': x_coordinates,
            'y': y_coordinates
        })

        self['g']=self['g'].astype('category')

        # Initiate pixel maps:
        self.pixel_maps = []
        self.stats = PointStatistics(self)

        if props is None:
            self.props = PropsDF(self)
        else:
            self.props= props

        self.graph = SpatialGraph(self)

        for pm in pixel_maps:
            if not type(pm) == PixelMap:
                self.pixel_maps.append(PixelMap(pm))
            else:
                self.pixel_maps.append(pm)

        self.synchronize = synchronize

        # Append scanpy data set, synchronize both:
        if scanpy is not None:
            self.scanpy = ScanpyDataFrame(self, scanpy)
            if self.synchronize:
                self.sync_scanpy()
        else:
            self.scanpy=None

        # self.obsm = {"spatial":np.array(self.coordinates).T}
        # self.obs = pd.DataFrame({'gene':self.g})
        self.uns={}

    @property
    def gene_ids(self):
        return self.gene_id

    @property
    def coordinates(self):
        return np.array([self.x,self.y]).T

    @property
    def counts(self):
        return self.stats['counts']

    @property
    def counts_sorted(self):
        return self.stats.counts[self.stats.count_indices]

    @property
    def genes_sorted(self):
        return self.genes[self.stats.count_indices]

    @property
    def genes(self):
        return self.stats.index

    @property
    def spatial(self):
        return SpatialIndexer(self)

    @property
    def background(self):
        if len(self.pixel_maps):
            return self.pixel_maps[0]
    @property
    def adata(self):
        if self.scanpy is not None:
            return self.scanpy.adata

    @property
    def X(self):
        return scipy.sparse.csc_matrix((np.ones(len(self.g),),(np.arange(len(self.g)),np.array(self.gene_ids).flatten())),
                        shape=(len(self.g),self.genes.shape[0],))

    @property
    def var(self):
        return pd.DataFrame(index=self.stats.genes)

    @property
    def obs(self):
        return  pd.DataFrame({'gene':self.g}).astype(str).astype('category')

    @property
    def obsm(self):
        return {"spatial":np.array(self.coordinates).T}

    def __getitem__(self, *arg):

        if (len(arg) == 1):

            if type(arg[0]) == str:

                return super().__getitem__(arg[0])

            if (type(arg[0]) == slice):
                new_data = super().iloc.__getitem__(arg)

            elif (type(arg[0]) == int):
                new_data = super().iloc.__getitem__(slice(arg[0], arg[0] + 1))

            elif isinstance(arg[0], pd.Series):
                # print(arg[0].values)
                new_data = super().iloc.__getitem__(arg[0].values)


            elif isinstance(arg[0], np.ndarray):
                new_data = super().iloc.__getitem__(arg[0])

            elif isinstance(arg[0], collections.Sequence):
                if all([a in self.keys() for a in arg[0]]):
                    return super().__getitem__(*arg)
                new_data = super().iloc.__getitem__(arg[0])

            if self.scanpy is not None:
                scanpy = self.scanpy.adata
                synchronize = self.scanpy.synchronize
            else:
                scanpy = None
                synchronize = None

            new_frame = SpatialData(new_data.g,
                                    new_data.x,
                                    new_data.y,
                                    self.pixel_maps,
                                    scanpy=scanpy,
                                    synchronize=synchronize,
                                    props=self.props.copy())

            new_prop_entries=self.props.loc[new_frame.genes]
            new_frame.props[new_prop_entries.columns]=new_prop_entries
            new_frame.props.sd = new_frame
            new_frame.props.drop(self.genes.symmetric_difference(new_frame.genes) ,inplace=True)

            if self.graph._umap is not None:
                new_frame.graph._umap = self.graph._umap[self.index.isin(new_frame.index)]
                
            return (new_frame)

        print('Reverting to generic Pandas.')
        return super().__getitem__(*arg)

    def sync_scanpy(self,
                    mRNA_threshold_sc=1,
                    mRNA_threshold_spatial=1,
                    verbose=False,
                    anndata=None):
        if anndata is None and self.scanpy is None:
            print('Please provide some scanpy data...')

        if anndata is not None:
            self.scanpy = ScanpyDataFrame(anndata)
        else:
            self.scanpy.synchronize()

    def get_id(self, gene_name):
        return int(self.stats.gene_ids[self.genes == gene_name])


    def scatter(self,
                c=None,
                color=None,
                legend  =None,
                axd=None,
                plot_bg=True,
                cmap='jet',
                scalebar=True,
                **kwargs):

        if axd is None:
            axd = plt.gca()

        handle_imshow=None
        handle_legend=None

        if self.background and plot_bg:
            handle_imshow=self.background.imshow(cmap='Greys', axd=axd)

        if c is None and color is None:
            c = self.props.project('c_genes')
            clrs=self.props.loc[self.genes].c_genes 
        else:
            cmap = get_cmap(cmap)    
            clrs = [cmap(f) for f in np.linspace(0,1,len(self.genes))]

        
        if legend:
            handles = [plt.scatter([],[],color=c) for c in clrs]
            handle_legend = plt.legend(handles,self.genes)


        # axd.set_title(gene)
        handle_scatter = axd.scatter(self.x,
                    self.y,
                    c=c,
                    color=color,
                    cmap=cmap,
                    **kwargs)

        if scalebar:
            self.add_scalebar(axd=axd)

        return handle_scatter,handle_imshow,handle_legend

    def add_scalebar(self, length=None, unit=r'$\mu$m',axd=None, color='w'):

        if axd is None:
            axd=plt.gca()
        
        x_,_x = axd.get_xlim()
        y_,_y = axd.get_ylim()

        if length is None:
            length = (_x-x_)*0.3
            decimals = np.ceil(np.log10(length))-1
            inter = length/10**decimals
            length = np.ceil(inter) * 10**decimals #int(np.around(_x-x_,1-int(np.log10((_x-x_)/2))))
        
        _x_ = _x-x_
        _y_ = _y-y_

        new_x_ = x_ + _x_/20
        new__x =  new_x_+length
        new_y = y_ + _y_/20

        scbar = plt.Line2D([0.1,0.9],[0.5,0.5],c='w', marker='|',linewidth=2,
            markeredgewidth=3,markersize=10,color=color)
        scbar.set_data([new_x_,new__x],[new_y,new_y])
        scbar.set_path_effects([PathEffects.withStroke(linewidth=4, foreground='k')])


        magnitudes=['nm',r'$\mu$m','mm','m','km']
        bar_label = f'{int(length%999)}{magnitudes[np.floor(np.log10(length)/3+1).astype(int)]}'

        sctxt = axd.text((new_x_+new__x)/2, (y_+_y_/15),bar_label, fontweight='bold', ha='center',color=color )

        axd.add_artist(scbar)
        sctxt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='k')])

        return scbar, sctxt


    def plot_bars(self, axis=None, **kwargs):
        if axis is None:
            axis = plt.subplot(111)
        axis.bar(np.arange(len(self.stats.counts)), self.counts_sorted,
                 **kwargs)
        axis.set_yscale('log')

        axis.set_xticks(
            np.arange(len(self.genes_sorted)),
            list(self.genes_sorted),
            )
        axis.tick_params(axis='x', rotation=90)
        axis.set_ylabel('molecule count')

    def plot_overview(self):
        
        colors = ('royalblue', 'goldenrod', 'red', 'lime')

        scatter_idcs = np.round(np.linspace(0,
                                            len(self.stats.counts) - 1,
                                            4)).astype(int)

        fig, axd = plt.subplot_mosaic(
            [['scatter_1', 'scatter_2', 'scatter_3', 'scatter_4'],
             ['bar', 'bar', 'bar', 'bar']],
            figsize=(11, 7),
            constrained_layout=True)

        self.plot_bars(
            axd['bar'],
            color=[
                colors[np.where(
                    scatter_idcs == i)[0][0]] if i in scatter_idcs else 'grey'
                for i in range(len(self.stats.counts))
            ])

        for i in range(4):
            idx = self.stats.count_indices[scatter_idcs[i]]
            gene = self.genes[idx]
            plot_name = 'scatter_' + str(i + 1)
            axd[plot_name].set_title(gene)
            axd[plot_name].scatter(self.x, self.y, color=(0.5, 0.5, 0.5, 0.1))
            axd[plot_name].scatter(self.x[self['gene_id'] == idx],
                                   self.y[self['gene_id'] == idx],
                                   color=colors[i],
                                   marker='.')

            axd[plot_name].set_xticks([], [])
            # if i>0:
            axd[plot_name].set_yticks([], [])

            con = ConnectionPatch(xyA=(scatter_idcs[i],
                                       self.stats.counts[idx]),
                                  coordsA=axd['bar'].transData,
                                  xyB=(np.mean(axd[plot_name].get_xlim()),
                                       axd[plot_name].get_ylim()[0]),
                                  coordsB=axd[plot_name].transData,
                                  color='white',
                                  linewidth=1,
                                  linestyle='dotted')
            fig.add_artist(con)

        plt.suptitle('Selected Expression Densities:', fontsize=18)

    def plot_radial_distribution(self, n_neighbors=30, **kwargs):
        # distances, _, _ = self.knn(n_neighbors=n_neighbors)
        self.graph.update_knn(n_neighbors=n_neighbors)
        distances = self.graph.distances
        plt.hist(distances[:, 1:n_neighbors].flatten(), **kwargs)

    def spatial_decomposition(
        self,
        mRNAs_center=None,
        mRNAs_neighbor=None,
        n_neighbors=10,
    ):

        if mRNAs_center is None:
            mRNAs_center = self.genes
        if mRNAs_neighbor is None:
            mRNAs_neighbor = self.genes

        self.graph.update_knn(n_neighbors=n_neighbors)
        neighbors = self.graph.neighbors
        # np.array(self.gene_ids)[neighbors]
        neighbor_classes = self.graph.neighbor_types

        pptx = []
        ppty = []
        clrs = []
        intensity = []

        out = np.zeros((30, 30, len(self.genes)))

        mask_center = np.logical_or.reduce(
            [neighbor_classes[:, 0] == self.get_id(m) for m in mRNAs_center])

        for i_n_neighbor in range(1, n_neighbors):

            mask_neighbor = np.logical_or.reduce([
                neighbor_classes[:, i_n_neighbor] == self.get_id(m)
                for m in mRNAs_neighbor
            ])

            mask_combined = np.logical_and(mask_center, mask_neighbor)

            for i_neighbor, n in enumerate(neighbors[mask_combined]):

                xs = np.array(self.x.iloc[n])
                ys = np.array(self.y.iloc[n])

                x_centered = xs - xs[0]
                y_centered = ys - ys[0]

                loc_neighbor = np.array(
                    (x_centered[i_n_neighbor], y_centered[i_n_neighbor]))
                loc_neighbor_normalized = loc_neighbor / (loc_neighbor **
                                                          2).sum()**0.5

                rotation_matrix = np.array(
                    [[loc_neighbor_normalized[1], -loc_neighbor_normalized[0]],
                     [loc_neighbor_normalized[0], loc_neighbor_normalized[1]]])

                rotated_spots = np.inner(
                    np.array([x_centered, y_centered]).T, rotation_matrix).T

                # we want to exclude the central and n_neighbor spots:
                mask = np.arange(rotated_spots.shape[1])
                mask = (mask > 1) & (mask != (i_n_neighbor))

                #         #         plt.scatter(rotated[0][mask],rotated[1][mask])

                pptx.append(rotated_spots[0][mask])
                ppty.append(rotated_spots[1][mask])
                clrs.append(self.gene_ids.iloc[n][mask])
                

        pptx = np.concatenate(pptx)
        ppty = np.concatenate(ppty)

        pptt = np.arctan(pptx / ppty)
        pptr = (pptx**2 + ppty**2)**0.5

        clrs = np.concatenate(clrs)
        
        scale = pptr.max()
        for i in range(len(mRNAs_neighbor)):
            mask = clrs==i
            out[(pptt[mask]/1.5*100).astype(int),(pptr[mask]/scale*100).astype(int),i]+=1

        plt.axhline(0)
        plt.axvline(0)
        plt.scatter(pptt, pptr, c=clrs, cmap='nipy_spectral', alpha=0.1, s=3)

        return (pptt, pptr, clrs,)

    def knn_clean(
        self,
        n_neighbors=10,
    ):
        # distances, indices, types = self.knn(n_neighbors=n_neighbors)
        self.graph.update_knn(n_neighbors=n_neighbors)
        types = self.graph.neighbor_types
        count_matrix = sparse.lil_matrix(
            (types.shape[0], self.genes.shape[0]))
        for i, t in enumerate(types):
            classes, counts = (np.unique(t[:n_neighbors], return_counts=True))
            count_matrix[i, classes] = counts / counts.sum()

        count_matrix = count_matrix.tocsr()

        count_matrix_log = count_matrix.copy()
        count_matrix_log.data = np.log(count_matrix.data)
        count_matrix_inv = count_matrix.copy()
        count_matrix_inv.data = 1 / (count_matrix.data)

        prototypes = np.zeros((len(self.genes), ) * 2)
        for i in range(prototypes.shape[0]):
            prototypes[i] = count_matrix[self.gene_ids == i].sum(0)
        prototypes /= prototypes.sum(0)

        Dkl = count_matrix.copy()

        for i in range(prototypes.shape[0]):
            inter = Dkl[self.gene_ids == i]
            inter.data = count_matrix[self.gene_ids == i].data * (np.log(
                (count_matrix_inv[self.gene_ids == i].multiply(
                    prototypes[i])).data))
            Dkl[self.gene_ids == i] = inter
        Dkl = -np.array(Dkl.sum(1)).flatten()
        Dkl[np.isinf(Dkl)] = 0

        return Dkl

    def scatter_celltype_affinities(self,
                                    adata,
                                    celltypes_1,
                                    celltypes_2=None):
        adata, sdata = synchronize(adata, self)

    def squidpy(self):
        # obs={"cluster":self.gene_id.astype('category')}
        obsm = {"spatial":np.array(self.coordinates)}
        # var= self.genes
        # self.obs = self.index
        # X = self.X #scipy.sparse.csc_matrix((np.ones(len(self.g),),(np.arange(len(self.g)),np.array(self.gene_ids).flatten())),
                        # shape=(len(self.g),self.genes.shape[0],))

        # sparse_representation = scipy.sparse.scr()
        # var = self.var #pd.DataFrame(index=self.genes)
        uns = self.uns.update({'Image':self.background})
        obs = pd.DataFrame({'gene':self.g})
        obs['gene']=obs['gene'].astype('category')
        return  anndata.AnnData(X=self.X,obs=obs,var=self.var,obsm=obsm)

    def save(self, path):
        pickle.dump({'sdata':self,'graph':self.graph,'props':self.props,}, open( path, "wb" ) )

## here starts plotting.py

def load(path):
    data = pickle.load(open( path, "rb" ))
    sdata=SpatialData(data['sdata']['g'],data['sdata']['x'],data['sdata']['y'])
    print(data['sdata'].columns)
    sdata.graph=data['graph']
    sdata.props=data['props']
    return sdata

def create_colorarray(sdata,values,cmap=None):
    if cmap is None:
        return values[sdata.gene_ids]





def hbar_compare(stat1,stat2,labels=None,text_display_threshold=0.02,c=None):


    genes_united=sorted(list(set(np.concatenate([stat1.index,stat2.index]))))[::-1]
    counts_1=[0]+[stat1.loc[i].counts if i in stat1.index else 0 for i in genes_united]
    counts_2=[0]+[stat2.loc[i].counts if i in stat2.index else 0 for i in genes_united]
    cum1 = np.cumsum(counts_1)/sum(counts_1)
    cum2 = np.cumsum(counts_2)/sum(counts_2)

    if c is None:
        c = [None]*len(cum1)

    for i in range(1,len(cum1)):

        bars = plt.bar([0,1],[cum1[i]-cum1[i-1],cum2[i]-cum2[i-1]],
                bottom=[cum1[i-1],cum2[i-1],], width=0.4,color=c[i-1])
        clr=bars.get_children()[0].get_facecolor()
        plt.plot((0.2,0.8),(cum1[i],cum2[i]),c='k')
        plt.fill_between((0.2,0.8),(cum1[i],cum2[i]),(cum1[i-1],cum2[i-1]),color=clr,alpha=0.2)
        
        if (counts_1[i]/sum(counts_1)>text_display_threshold) or \
        (counts_2[i]/sum(counts_2)>text_display_threshold): 
            plt.text(0.5,(cum1[i]+cum1[i-1]+cum2[i]+cum2[i-1])/4,
                    genes_united[i-1],ha='center',)  

    if labels is not None:
        plt.xticks((0,1),labels) 

def sorted_bar_compare(stat1,stat2,kwargs1={},kwargs2={}):
    categories_1 = (stat1.index)
    counts_1 = (np.array(stat1.counts).flatten())
    counts_1_idcs = np.argsort(counts_1)
    # count_ratios = np.log(self.determine_gains())
    # count_ratios -= count_ratios.min()
    # count_ratios /= count_ratios.max()

    ax1 = plt.subplot(311)
    ax1.set_title('compared molecule counts:')
    ax1.bar(np.arange(len(counts_1)), counts_1[counts_1_idcs], color='grey',**kwargs1)
    # ax1.set_ylabel('log(count) scRNAseq')
    ax1.set_xticks(np.arange(len(categories_1)),
                categories_1[counts_1_idcs],
                )
    ax1.tick_params(axis='x', rotation=90)
    ax1.set_yscale('log')

    ax2 = plt.subplot(312)
    for i, gene in enumerate(categories_1[counts_1_idcs]):
        if gene in stat2.index:
            plt.plot(
                [i, stat2.count_ranks[gene]],
                [1, 0],
            )
    plt.axis('off')
    ax2.set_ylabel(' ')

    ax3 = plt.subplot(313)
    ax3.bar(np.arange(len(stat2)), stat2.counts[stat2.count_indices], color='grey',**kwargs2)

    ax3.set_xticks(np.arange(len(stat2.index)),
                stat2.index[stat2.count_indices],
                rotation=90)
    ax3.set_yscale('log')
    ax3.invert_yaxis()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    # ax3.set_ylabel('log(count) spatial')
    return(ax1,ax2,ax3)

   
   