import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('summary_2023-03-19_09_09_08.csv', low_memory=False)

df['total_fix_mode'] = df[['fix_mode0','fix_mode1','fix_mode2', 'fix_mode3','fix_mode4', 'fix_mode5', 'fix_mode6']].sum(axis=1)
#df['total_fix_mode'] = df[['fix_mode0', 'fix_mode1', 'fix_mode2', 'fix_mode3', 'fix_mode4', 'fix_mode5', 'fix_mode6']].sum(axis=1)
#dev_id,station_dev_id,station_id,distance,fix_mode0,fix_mode1,fix_mode2,fix_mode3,fix_mode4,fix_mode5,fix_mode6,total_epoch,g_epoch,r_epoch,e_epoch,c_epoch,sat,slip,tec,scinti,lat,lng,start,end,vrs
# 新增一列fix_count为fix_mode3和fix_mode6之间的较大值
df['fix_count'] = df[['fix_mode3', 'fix_mode6']].max(axis=1)
df['fix_rate'] = df['fix_count'] / df['total_fix_mode']
df['rtcm_rate'] = (df['g_epoch'] ) / (df['total_epoch'] + 1)
#df['fix_state'] = df['fix_rate'].apply(lambda x: 1 if x > 0.90 else 0)
df['fix_state'] = df.apply(lambda x: 1 if x['fix_rate'] > 0.95 and x['fix_mode5'] < 10 and x['fix_mode4'] < 10 else 0, axis=1)
df['distance'] = df['distance'] / 1000

def showRates(df):
# Fix Rate Distribution
    plt.hist(df['fix_state'])
    plt.title('Fix Rate Distribution')
    plt.xlabel('Fix Rate')
    plt.ylabel('Count')
    plt.show()
    
    # RTCM Rate Distribution
    plt.hist(df['rtcm_rate'], bins=20)
    plt.title('RTCM Rate Distribution')
    plt.xlabel('RTCM Rate')
    plt.ylabel('Count')
    plt.show()
    
    # Session Time Distribution
    plt.hist(df['session_time'], bins=20)
    plt.title('Session Time Distribution')
    plt.xlabel('Session Time')
    plt.ylabel('Count')
    plt.show()


def loadhainan(df):
    df = df[(df['lng'] > 108.130085) & (df['lng']<111.662365)]
    df = df[(df['lat'] > 18.232757) & (df['lat']< 20.155466)]
    empty_df = pd.DataFrame()
    empty_df['time'] =  pd.to_datetime(df['start'], unit='ms')
    empty_df['dev_id'] = df['dev_id']
    empty_df['scinti'] = df['scinti']
    empty_df['fix_rate'] = df['fix_rate']
    empty_df['ts'] = df['start']
    empty_df.to_csv('filtered_data.csv', index=False)

def loadhainan2(df):
    df = filters(df)
    empty_df = pd.DataFrame()
    empty_df['scinti'] = df['scinti']
    empty_df['fix_rate'] = df['fix_rate']
    empty_df['fix_state'] = df['fix_state']
    empty_df['rtcm_rate'] = df['rtcm_rate']
    empty_df['scinti'] = df['scinti']
    empty_df['tec'] = df['tec']
    empty_df['distance'] = df['distance']
    empty_df.to_csv('filtered_data2.csv', index=False)

def loadDistanceTecGrid(df):
    df = filters(df)
    x = []
    y = []
    z = []
    for dist in [0, 10, 20, 30, 40, 50, 60]:
        for tec in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120]:
            #df_filtered = df[(df['distance'] >= dist) & (df['distance'] < dist+10) & (df['tec'] >= tec) & (df['tec'] < tec + 10)]
            df_filtered = df[(df['distance'] >= dist) & (df['tec'] >= tec)]
            totalcount = df_filtered.shape[0]
            df_filtered = df_filtered[df_filtered['fix_state'] == 1]
            fixcount2 = df_filtered.shape[0]
            if (totalcount == 0):
                continue
            x.append(dist)
            y.append(tec)
            z.append(fixcount2 * 1.0 / totalcount)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, z, )
    ax.set_xlabel('Distance')
    ax.set_ylabel('Tec')
    ax.set_zlabel('Rate')
    plt.show()       

def filters(df):
    df = df[df['distance'] > 1]
    df = df[df['tec'] > 1]
    df = df[df['distance'] < 70]
    #df = df[df['session_time'] > 40]
    #df = df[df['rtcm_rate'] > 0.9]
    #df = df[df['fix_count'] > 10]
    df = df[df['fix_mode1'] == 0]
    df = df[df['fix_mode2'] == 0]
    df = df[df['fix_mode3'] == 0]
    #df = df[df['scinti'] == 0]
    return df

# 写一个函数，接受df为参数，以distance为横轴，tec为纵轴，fix_state=1为红点，fix_state=0为蓝点生成散点图
def scatter_plot(df):
    df = filters(df)
    plt.ylabel('Distance')
    plt.xlabel('TEC')
    plt.scatter(df[df['fix_state']==1]['tec'],df[df['fix_state']==1]['distance'], c='red', s=0.01)
    plt.show()
    plt.ylabel('Distance')
    plt.xlabel('TEC')
    plt.scatter(df[df['fix_state']==0]['tec'],df[df['fix_state']==0]['distance'], c='blue', s=0.01)
    plt.show()

def showRtcmFixRate(df):
    fig, ax = plt.subplots()
    ax.scatter(df['rtcm_rate'], df['fix_rate'])
    ax.set_xlabel('rtcm_rate')
    ax.set_ylabel('fix_rate')
    plt.show()
    
def showRtcmFixBar(df):
    grouped = df.groupby(['fix_state', pd.cut(df['rtcm_rate'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])]).size().unstack()
# 绘制堆积条形图
    ax = grouped.plot(kind='bar', stacked=True, figsize=(10, 6))
    ax.set_title('Fix State and RTCM Rate', fontsize=16)
    ax.set_xlabel('RTCM Rate', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(['0', '1'])
    plt.show()