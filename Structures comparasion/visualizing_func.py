


from netCDF4 import Dataset
import netCDF4 as nc
import os
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
from mpl_toolkits.basemap import Basemap
import urllib
import numpy as np
import datetime
import calendar as cal
from numpy import array
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

def visual(structures,year,sektory)
    
    flat_list = [item for sublist in STRUCTURES for item in sublist]
    unique_data = [list(x) for x in set(tuple(x) for x in flat_list)]
    thick_dict = {tuple(thick[0:]):1 for thick in unique_data}
    
    for p in range(0,len(structures)):

        titletime = 'Sektory'                            # change date

        if p < 9:

            data = nc.Dataset(f'Arc_{year}0{p+1}15_res3.125_pyres.nc')                      # choose file
        else:
            data = nc.Dataset(f'Arc_{year}{p+1}15_res3.125_pyres.nc') 
        ice = data.variables['sea_ice_concentration'][:]
        lat = data.variables['lat'][:]          # от 61.64559174 до 81.95233154
        lon = data.variables['lon'][:]          # от 53.12836075 до 83.53308868  
        data.close()

        ### Convert to fraction
        ice = np.asarray(np.squeeze(ice/100.))

        ### Set missing data
        ice[np.where(ice <= 0.2)] = np.nan              # нужно поставить правильное ограничение
        ice[np.where(ice > .999)] = .95

        ### Define parameters (dark)

        def setcolor(x, color):
             for m in x:
                for t in x[m][1]:
                     t.set_color(color)

        ### Plot Sea Ice Concentration
        fig = plt.figure(figsize=(25,12))  #fig = plt.figure(figsize=(15,7))
        ax = fig.add_subplot(111)

        m = Basemap(projection='npstere',boundinglat=30,lon_0=270,resolution='l',round=True)   # поработать с приближением и локацией карты
        m.drawcoastlines(color = 'k',linewidth=0.7)
        m.drawcountries(color='k',linewidth=0.5)
        m.fillcontinents(color='lightgrey')
        m.drawmapboundary(color='black')

        parallels = np.arange(60,90,5)                                                         # вот тут нужно правильно указать параллели, иначе кидает ошибку (покак это вариант работает, но не появляется лед)
        meridians = np.arange(-180,180,30)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[0,0,0,0],linewidth=0.5,color='black')
        par=m.drawmeridians(meridians,labels=[1,1,0,0],linewidth=0.5,fontsize=6,color='black')
        setcolor(par,'black')


        #from matplotlib import ticker, cm


        cs = m.contourf(lon,lat,ice[:,:],np.arange(0.2,1.04,.05),extend='min', latlon=True, alpha=0.7)                   # на этом моменте должен появляться лед, но он не появляется
        cs2 = m.contour(lon,lat,ice[:,:],np.arange(0.2,0.5,0.1),latlon=True,colors='blue',linewidths=1.5, alpha=0.6)     
        cs.set_cmap('Blues')
        for oblast in sektory:
            cs4 = m.contourf(lon,lat,oblast[:,:],np.arange(.05,1.04,0.2),latlon=True,extend='max',alpha=0.4) 
            cs4.set_cmap('Greens')


        for ddf in strucutures[p]:
            thick_dict[tuple(ddf[0:])]=thick_dict[tuple(ddf[0:])]+0.5
            xqq, yqq = m(lon_coord[int(ddf[1])], lat_coord[int(ddf[1])])

            xqq2, yqq2 = m(lon_coord[int(ddf[0])], lat_coord[int(ddf[0])])

            plt.annotate('', xy=(xqq, yqq),  xycoords='data',
                        xytext=(xqq2, yqq2), textcoords='data',
                        color='r',
                        arrowprops=dict(arrowstyle="->, head_length=0.3,  head_width=0.3", color='darkred',lw=cars_dict[tuple(ddf[0:])])
                        )


        #lons= [-48.26,-34.82,-23.119999, -15.139999,-12.69]
        #lats=[43.71,44.11, 42.71,41.02,41.39]

        x, y = m(lon_coord, lat_coord)

        m.scatter(x, y, marker='D',color='darkgreen',s=10)

        for re in range(37):
            xre, yre = m(lon_coord[re], lat_coord[re])
            plt.text(xre, yre, re,fontsize=12,fontweight='bold', ha='left',va='bottom',color='w')

        #plt.text(lon_coord[0], lat_coord[0], 'Lagos',fontsize=120,fontweight='bold',
        #                    ha='left',va='bottom',color='k')


        cbar = m.colorbar(cs,drawedges=True,location='right',pad = 0.55)
        cbar.set_label('Sea Ice Concentration',fontsize=13)
        cbar.ax.tick_params(axis='y', size=.6)

        fig.suptitle('Sea Ice Concentration',      
                     fontsize=16,color='black')

        fig.subplots_adjust(top=0.88)

        ### Save figure 
        plt.savefig(f'g{year}{p}.png') # folder for graphs https://drive.google.com/drive/u/0/folders/1Ps5JIpfp1AQS21-JVZGg851Dae_po3SJ
        print(p)
        plt.show()

