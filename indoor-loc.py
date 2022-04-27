import networkx as nx
import os
import sys
sys.path.append('graph-neural-networks/')
import torch
import alegnn.modules.architectures as architectures
import alegnn.modules.loc_gnn as loc_gnn 
import alegnn.utils.graphML as graphML
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, mean_squared_error
import scipy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import argparse
mpl.rcParams.update(mpl.rcParamsDefault)

NUM_BUILDINGS = 3
NUM_FLOORS = 5
BLD_FLR_COUNTS = {0: 4, 1: 4, 2: 5}
RSSI_OFFSET = 105
NO_SIGNAL_INDICATOR = 100
TOTAL_NUM_APS = 520

def generate_graph(features, selected_aps, rssi_threshold=10):
    df_features = pd.DataFrame(features, columns=selected_aps)
    df_G = pd.DataFrame(columns=['source', 'target', 'weight']) 

    for i in selected_aps:
        max_rssi = df_features[i].max()
        curr_ap_neighbours = df_features[df_features[i]  > (max_rssi - rssi_threshold)]
        curr_ap_neighbours = curr_ap_neighbours.drop(i, axis=1) 
        
        for k, v in curr_ap_neighbours.mean().items():
            df_G = df_G.append({'source': i, 'target': k, 'weight': v}, ignore_index=True)
    
    G = nx.from_pandas_edgelist(df_G, source='source', target='target', edge_attr='weight') 
    W = nx.to_numpy_array(G)
    np.fill_diagonal(W, 0)
    (w, v) = scipy.sparse.linalg.eigs(W, k=1, which='LM')
    W = W / np.abs(w[0])

    return G, W

def data_analysis(X_train, selected_aps):
    X_train = X_train.replace(to_replace=0, value=np.nan)
    X_stack = X_train.stack(dropna=False)
    plt.grid()
    plt.xlabel('RSSI (scaled)')
    plt.ylabel('Frequency')
    sns.distplot(X_stack.dropna(), kde=False)
    
    waps_in_range = (X_train.notnull().sum(axis=1))
    fig, ax = plt.subplots(1, 1)
    plt.grid()
    sns.distplot(waps_in_range, ax=ax, kde=False, color='#e4cfff')
    ax.set_xlabel("Number of APs in range")
    plt.show()

def train_model(model, train_data, indexes, batch_size=32, n_epochs=100, learning_rate=0.005, weight_decay=1e-2, plot_loss=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False)
    loss_vals = []
    loss_funcs = {
        'cls-bld': torch.nn.CrossEntropyLoss(), 
        'cls-flr': torch.nn.CrossEntropyLoss(), 
        'reg-coords': torch.nn.MSELoss(), 
        'cls-bld-flr': torch.nn.CrossEntropyLoss(), 
        'cls-refloc': torch.nn.CrossEntropyLoss()
    }

    for epoch in range(n_epochs):
        epoch_loss = []
        for x_batch, y_batch in train_loader:
            if y_batch.shape[0] == batch_size:
                model.zero_grad()
                results = model(x_batch)
                losses = [loss_funcs[mlp_type](results[mlp_type], y_batch[:, indexes[mlp_type]].reshape(batch_size if 'cls' in mlp_type else (batch_size, 2)).type(torch.long if 'cls' in mlp_type else torch.float)) for mlp_type in results]
                total_loss = sum(losses)
                total_loss.backward()
                optimizer.step()
                epoch_loss.append(total_loss.item())
            else:
                break
        loss_vals.append(sum(epoch_loss) / len(epoch_loss))
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'[INFO] Epoch {epoch + 1}. Loss:', sum(epoch_loss)/len(epoch_loss))
            
    if plot_loss:
        plt.title('Learning curve')
        plt.plot(loss_vals)
        plt.show()

    return (model, loss_vals)

def generate_predictions(model, test_data):
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)
    for x_batch, y_batch in test_loader:
        predicted = model(x_batch)
    return predicted

def evaluate(y_pred, y_test, indexes, encoders, val=False, conf_mat=False, refloc_params=None, sigma=.5, k=8, verbose=True):
    results = {}
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():  
        for mlp_type in y_pred:
            results[mlp_type] = {}
            if (not val and mlp_type != 'cls-refloc') or val:
                actual = y_test[:, indexes[mlp_type]]
                predicted = y_pred[mlp_type]
                
                if 'cls' in mlp_type:
                    actual = encoders[mlp_type].inverse_transform(actual.reshape(-1, 1)).reshape(-1,)
                    predicted = np.array(torch.argmax(softmax(predicted), axis=1)).reshape(-1, 1)
                    predicted = encoders[mlp_type].inverse_transform(predicted).reshape(-1,)
                    
                    if mlp_type == 'cls-refloc':
                        acc = 0
                        for i in range(len(predicted)):
                            if type(predicted[i]) != int or type(actual[i]) != int:
                                print(predicted[i], type(predicted[i]), actual[i], type(actual[i]))
                            acc += predicted[i] == actual[i]
                        accuracy = acc / len(predicted)
                    else:
                        accuracy = accuracy_score(actual, predicted)
                    results[mlp_type] = {'accuracy': accuracy}
                else:
                    actual = encoders[mlp_type].inverse_transform(actual)
                    predicted = encoders[mlp_type].inverse_transform(predicted.cpu().detach().numpy())
                    diffs = np.sqrt(np.sum((predicted - actual)**2, axis=1))
                    results[mlp_type] = {'rmse': np.mean(diffs), 'median': np.median(diffs)}
                
        if 'cls-bld-flr' in mlp_types:
            res = extract_bld_flr(y_pred['cls-bld-flr'], y_test[:, indexes['cls-bld-flr']], encoders['cls-bld-flr'], conf_mat=conf_mat, verbose=verbose)
            results['cls-bld-flr'] = {**results['cls-bld-flr'], **res, **results['cls-bld-flr']}

        if 'cls-refloc' in mlp_types and refloc_params:
            res = evaluate_reflocs(y_pred, refloc_params[0], encoders, refloc_params[1], sigma=sigma, k=k, verbose=verbose)
            results['cls-refloc'] = res

    if verbose:
        for mlp_type in results:
            if mlp_type != 'cls-refloc' and mlp_type != 'cls-bld-flr':
                for metric in results[mlp_type]:
                    print(task_names[mlp_type], metric + ':', results[mlp_type][metric])

    return results

def evaluate_reflocs(y_pred, df_train, encoders, y_test, sigma=.5, k=8, verbose=True): # top 5 locations; assume building and floor predictions were passed; y_test should contain latitude and longitude
    with torch.no_grad(): 
        softmax = torch.nn.Softmax(dim=1)
        idxs = np.argpartition(y_pred['cls-refloc'], -k)[:, -k:]
        bld_flrs_predicted = encoders['cls-bld-flr'].inverse_transform(np.array(torch.argmax(softmax(y_pred['cls-bld-flr']), axis=1)).reshape(-1, 1)).reshape(-1,)
        bld_flrs_actual = y_test['BLD_FLR'].values.reshape(-1,)
        weighted_predicted_locs = np.zeros((len(bld_flrs_actual), 2))
        weighted_predicted_locs[:, 0] = None 
        weighted_predicted_locs[:, 1] = None

        predicted_locs = np.zeros((len(bld_flrs_actual), 2))
        predicted_locs[:, 0] = None 
        predicted_locs[:, 1] = None
        correct = []
        total = 0
        mean = True
        threshold = sigma * np.amax(np.array(y_pred['cls-refloc']), axis=1)
        for i in range(len(bld_flrs_actual)):
            if bld_flrs_actual[i] == bld_flrs_predicted[i]:
                weights = []
                coords = []
                # calculate positioning error
                for j in idxs[i]:  
                    reflocid = encoders['cls-refloc'].inverse_transform([[j]]).reshape(-1,)[0]
                    rows = df_train[(df_train['BLD_FLR'] == bld_flrs_predicted[i]) & (df_train['REFLOC'] == reflocid)].reset_index(drop=True)
                    if not rows.empty and y_pred['cls-refloc'][i][j] > threshold[i]:
                        c = rows[['LATITUDE', 'LONGITUDE']].mean(axis=0) if mean else rows.loc[0, ['LATITUDE', 'LONGITUDE']]
                        coords.append(c.values)
                        weights.append([float(y_pred['cls-refloc'][i][j]), float(y_pred['cls-refloc'][i][j])])
                    
                if len(coords) > 0:
                    coords = np.array(coords)
                    weights = np.array(weights)
                    weighted_predicted_locs[i] = [np.average(coords[:, 0], weights=weights[:, 0]), np.average(coords[:, 1], weights=weights[:, 1])]
                    predicted_locs[i] = [np.mean(coords[:, 0]), np.mean(coords[:, 1])]
                    correct.append(i)
                total += 1
        
        ref_hit_rate = len(correct) / total
        weighted_diffs = np.sqrt(np.sum((weighted_predicted_locs[correct] - y_test.loc[correct, ['LATITUDE', 'LONGITUDE']].values)**2, axis=1))
        diffs = np.sqrt(np.sum((predicted_locs[correct] - y_test.loc[correct, ['LATITUDE', 'LONGITUDE']].values)**2, axis=1))
        wrmse = np.mean(weighted_diffs)
        rmse = np.mean(diffs)
        wmedian = np.median(weighted_diffs)
        median = np.median(diffs)

        if verbose:
            print('% of samples for which a reference location was found:', ref_hit_rate)
            print('RMSE (weighted):', wrmse)
            print('RMSE (equal weight):', rmse)
            print('Median error (weighted):', wmedian)
            print('Median error (equal weight):', median)

    return {'weighted centroid RMSE': wrmse, 'centroid RMSE': rmse, 'weighted centroid median': wmedian, 'centroid median': median, 'reference location hit rate': ref_hit_rate}
    
def plot_confusion_matrix(predicted, actual):
    cf_matrix = confusion_matrix(actual, predicted) 
    cf_matrix = cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cf_matrix, fmt='.2f', annot=True, cmap='Blues')

    ax.set_title('Confusion Matrix');
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values');

    ax.xaxis.set_ticklabels(sorted(set(predicted.reshape(-1,))))
    ax.yaxis.set_ticklabels(sorted(set(actual.reshape(-1,))))

    plt.show()

def build_model(curr_params, W, num_aps, output_lens, mlp_types):
    if curr_params['gnn_type'] == 'selection':
        gnn_model = loc_gnn.SelectionGNN(
            dimNodeSignals=[1] + ([curr_params['num_node_signals']] * curr_params['num_gnn_hidden']),
            nFilterTaps=[curr_params['num_filter_taps']] * curr_params['num_gnn_hidden'], 
            bias=True, 
            nonlinearity=curr_params['activation_func'], 
            nSelectedNodes=[num_aps] * curr_params['num_gnn_hidden'], 
            poolingFunction=curr_params['pooling_func'], 
            poolingSize=[curr_params['pooling_size']] * curr_params['num_gnn_hidden'], 
            dimLayersMLPs={key: ([output_lens[key]] if curr_params['mlp-hidden'] == 0 else [curr_params['mlp-hidden']] * curr_params['num_mlp_hidden'] + [output_lens[key]]) for key in mlp_types}, 
            GSO=torch.from_numpy(W).float(),
            dropout=curr_params['dropout']
        )
    elif curr_params['gnn_type'] == 'attention':
        gnn_model = loc_gnn.GraphConvolutionAttentionNetwork(
            dimNodeSignals=[1] + [curr_params['num_node_signals']] * curr_params['num_gnn_hidden'], 
            nFilterTaps=[curr_params['num_filter_taps']] * curr_params['num_gnn_hidden'], 
            nAttentionHeads=[curr_params['attention_heads']] * curr_params['num_gnn_hidden'], 
            bias=True,
            nonlinearity=curr_params['activation_func'],
            nSelectedNodes=[num_aps] * curr_params['num_gnn_hidden'], 
            poolingFunction=curr_params['pooling_func'], 
            poolingSize=[curr_params['pooling_size']] * curr_params['num_gnn_hidden'],
            dimLayersMLPs={key: ([curr_params['mlp-hidden']] * curr_params['num_mlp_hidden'] + [output_lens[key]] if curr_params['mlp-hidden'] > 0 else [output_lens[key]]) for key in mlp_types},
            GSO=torch.from_numpy(W).float(),
            dropout=curr_params['dropout']
        )
    return gnn_model

def prepare_datasets(X_train, y_train, X_test, y_test, input_size, output_size, reflocs=True, val=False):
    X_train = np.reshape(X_train, (X_train.shape[0], 1, input_size))
    X_test = np.reshape(X_test,(X_test.shape[0], 1, input_size))
    y_train = np.reshape(y_train, (y_train.shape[0], output_size))
    y_test = np.reshape(y_test, (y_test.shape[0], output_size - (1 if reflocs and not val else 0)))

    train_data = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    return train_data, test_data

def find_best_params(X_train, y_train, selected_aps, output_lens, mlp_types, indexes, gnn_type, num_aps, output_size, k=3, refloc_params=None, epochs=20, plot_loss=False, params=None):
    kf = KFold(n_splits=k)
    learning_rates = [1e-3]
    batch_sizes = [64, 128] 
    weight_decays = [1e-4]
    num_filters = [3, 5]
    rssi_thresholds = [5, 10]
    mlp_hidden = [32, 16]
    num_mlp_hidden = [1, 2]
    num_gnn_hidden = [3]
    num_node_signals = [20, 40]
    attention_heads = [1]
    pooling_funcs = [graphML.NoPool] # max pooling takes too long
    pooling_sizes = [1]
    dropout = [True, False]
    activation_funcs = [torch.nn.ReLU, torch.nn.Tanh] if gnn_type == 'selection' else [torch.nn.functional.relu, torch.nn.functional.tanh]
    gnn_types = [gnn_type]

    param_grid = {
        'learning_rate': learning_rates, 
        'batch_size': batch_sizes, 
        'weight_decay': weight_decays, 
        'num_filter_taps': num_filter_taps, 
        'rssi_threshold': rssi_thresholds, 
        'mlp-hidden': mlp_hidden, 
        'num_node_signals': num_node_signals, 
        'attention_heads': attention_heads, 
        'pooling_func': pooling_funcs, 
        'activation_func': activation_funcs, 
        'pooling_size': pooling_sizes,
        'num_mlp_hidden': num_mlp_hidden,
        'dropout': dropout,
        'num_gnn_hidden': num_gnn_hidden,
        'gnn_type': gnn_types
    }
    best_error = float('inf')
    best_params = None

    labels = [p for p in param_grid if len(param_grid[p]) > 1]

    for curr_params in list(ParameterGrid(param_grid)):
        curr_error = 0
        loss_vals = np.zeros((epochs,))
        print("[INFO] Trying parameters:", curr_params)
        for train_idxs, val_idxs in kf.split(X_train):
            X_train_fold, X_val_fold = X_train[train_idxs], X_train[val_idxs]
            y_train_fold, y_val_fold = y_train[train_idxs], y_train[val_idxs]

            Graph, W = generate_graph(X_train_fold, selected_aps, rssi_threshold=curr_params['rssi_threshold']) 
            
            gnn_model = build_model(curr_params, W, num_aps, output_lens, mlp_types)

            train_data, val_data = prepare_datasets(X_train_fold, y_train_fold, X_val_fold, y_val_fold, num_aps, output_size, reflocs='cls-refloc' in mlp_types, val=True)
            
            gnn_model, lv = train_model(gnn_model, train_data, indexes={key: indexes[key] for key in mlp_types}, n_epochs=epochs, learning_rate=curr_params['learning_rate'], batch_size=curr_params['batch_size'], weight_decay=curr_params['weight_decay'])
            loss_vals = (loss_vals + lv) / k
            
            y_pred = generate_predictions(gnn_model, val_data)
            results = evaluate(y_pred, y_val_fold.reshape(y_val_fold.shape[0], output_size), indexes, encoders, val=True)
            
            refloc_error = None
            if 'cls-refloc' in mlp_types and refloc_params is not None:
                refloc_error = evaluate_reflocs(y_pred, refloc_params[0], encoders, refloc_params[0].iloc[val_idxs, :].reset_index(drop=True))['weighted centroid RMSE']
            curr_error += (sum([(results[mlp_type]['rmse'] if not 'cls' in mlp_type else (1 - results[mlp_type]['accuracy']) if mlp_type != 'cls-refloc' else 0) for mlp_type in mlp_types]) + (0 if refloc_error is None else refloc_error)) / k
            print('One cross-validation fold complete. Results:', results)
        
        if plot_loss:
            plt.plot(loss_vals, label='; '.join([str(key) + ': ' + str(curr_params[key]) for key in labels]))
            
        if curr_error < best_error:
            best_error = curr_error
            best_params = curr_params

        print("Total error: ", curr_error) 
        print('-' * 50)

    print("----------- BEST PARAMS --------------")
    print("params: ", best_params)
    print("Total error: ", best_error) 
    print('-' * 50)

    if plot_loss:
        sns.set_theme()
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    return best_params

def extract_bld_flr(predicted, actual, encoder, conf_mat=False, verbose=True):
    softmax = torch.nn.Softmax(dim=1)
    actual = encoder.inverse_transform(actual.reshape(-1, 1)) 
    predicted = np.array(torch.argmax(softmax(predicted), axis=1)).reshape(-1, 1)
    predicted = encoder.inverse_transform(predicted)

    temp = np.array([[x[0], x[1]] for x in actual.reshape(-1,)]).astype(int)
    flr_actual = temp[:, 1]
    bld_actual = temp[:, 0]

    temp = np.array([[x[0], x[1]] for x in predicted.reshape(-1,)]).astype(int)
    flr_predicted = temp[:, 1]
    bld_predicted = temp[:, 0]

    results = {'building hit rate': accuracy_score(bld_actual, bld_predicted), 'floor hit rate': accuracy_score(flr_actual, flr_predicted), 'breakdown': {}}
    
    sep = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0], 2: [0, 0, 0, 0]}
    for i in range(len(bld_actual)):
        sep[bld_actual[i]][0] += bld_actual[i] == bld_predicted[i]
        sep[bld_actual[i]][1] += flr_actual[i] == flr_predicted[i]
        sep[bld_actual[i]][2] += 1
        sep[bld_actual[i]][3] += 1

    for bld in sep:
        results['breakdown'][bld] = {'building hit rate': sep[bld][0] / sep[bld][2], 'floor hit rate': sep[bld][1] / sep[bld][3], 'breakdown': {}}

    sep = {}
    for i in range(len(bld_actual)):
        if bld_actual[i] not in sep:
            sep[bld_actual[i]] = {}
        if flr_actual[i] not in sep[bld_actual[i]]:
            sep[bld_actual[i]][flr_actual[i]] = [0, 0]
        
        sep[bld_actual[i]][flr_actual[i]][0] += flr_actual[i] == flr_predicted[i]
        sep[bld_actual[i]][flr_actual[i]][1] += 1

    for bld in sorted(sep.keys()):
        for flr in sorted(sep[bld].keys()):
            results['breakdown'][bld]['breakdown'][flr] = sep[bld][flr][0] / sep[bld][flr][1]

    if verbose:
        print('Overall building hit rate:', results['building hit rate'])
        print('Overall floor hit rate:', results['floor hit rate'])
        for bld in results['breakdown']:
            print('Building', bld)
            print('Building hit rate:', results['breakdown'][bld]['building hit rate'], 'floor hit rate:', results['breakdown'][bld]['floor hit rate'])
            for flr in results['breakdown'][bld]['breakdown']:
                print(f'\tFloor {flr}. Hit rate: {results["breakdown"][bld]["breakdown"][flr]}')

    if conf_mat:
        plot_confusion_matrix(predicted, actual)

    return results

def load_data(training_dataset_path, test_dataset_path, mlp_types, mlp_col_dict, min_std=5, visualise=False):
    df_train = pd.read_csv(training_dataset_path)
    df_train.loc[:, 'BLD_FLR'] = df_train['BUILDINGID'].astype(str) + df_train['FLOOR'].astype(str)
    df_train.loc[:, 'REFLOC'] = df_train.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1)
    buildings = df_train['BUILDINGID'].unique()
    floors = df_train['FLOOR'].unique()
    for bld in buildings:
        for flr in floors:
            f = (df_train['BUILDINGID'] == bld) & (df_train['FLOOR'] == flr)
            _, idx = np.unique(df_train.loc[f, 'REFLOC'], return_inverse=True)
            df_train.loc[f, 'REFLOC'] = idx 
    
    df_test = pd.read_csv(test_dataset_path)
    df_test.loc[:, 'BLD_FLR'] = df_test['BUILDINGID'].astype(str) + df_test['FLOOR'].astype(str)
    
    
    label_names = ['LATITUDE', 'LONGITUDE'] if 'reg-coords' in mlp_types else []
    for mlp_type in mlp_types:
        if mlp_type != 'reg-coords':
            label_names.append(mlp_col_dict[mlp_type])
    
    df_X_train = df_train.iloc[:, :TOTAL_NUM_APS]
    df_X_train.values[df_X_train.values == NO_SIGNAL_INDICATOR] = -RSSI_OFFSET
    df_X_test = df_test.iloc[:, :TOTAL_NUM_APS] 
    df_X_test.values[df_X_test.values == NO_SIGNAL_INDICATOR] = -RSSI_OFFSET
    
    strongest_signal = (df_X_train.describe().iloc[2] > min_std)
    selected_aps = strongest_signal.index[strongest_signal.values]
    
    if visualise:
        sns.set_theme()
        sns.countplot(y="FLOOR", hue="BUILDINGID", data=df_train, orient="h")
        plt.show()
        data_analysis(df_X_train + RSSI_OFFSET, selected_aps)
        
    df_X_train = df_X_train[selected_aps] + RSSI_OFFSET
    df_X_test = df_X_test[selected_aps] + RSSI_OFFSET 
    df_y_train = df_train[label_names]
    df_y_test = df_test[[i for i in label_names if i != 'REFLOC']]
    num_aps = len(selected_aps)

    bld_encoder = OrdinalEncoder(categories=[buildings], dtype=int)
    flr_encoder = OrdinalEncoder(categories=[[i for i in range(max([v for k, v in BLD_FLR_COUNTS.items() if k in buildings]))]], dtype=int)
    bld_flr_encoder = OrdinalEncoder(categories=[df_train['BLD_FLR'].unique()])
    refloc_encoder = OrdinalEncoder(categories=[df_train['REFLOC'].unique()])
    coord_scaler = StandardScaler()
    rssi_scaler = StandardScaler()
    encoders = {'cls-bld-flr': bld_encoder, 'cls-flr': flr_encoder, 'reg-coords': coord_scaler, 'cls-bld-flr': bld_flr_encoder, 'cls-refloc': refloc_encoder}
    output_lens = {}
    indexes = {}
    if 'reg-coords' in mlp_types:
        output_lens = {'reg-coords': 2}
        indexes = {'reg-coords': [label_names.index('LATITUDE'), label_names.index('LONGITUDE')]}
        df_y_train.loc[:, ['LATITUDE', 'LONGITUDE']] = coord_scaler.fit_transform(df_y_train[['LATITUDE', 'LONGITUDE']])
        df_y_test.loc[:, ['LATITUDE', 'LONGITUDE']] = coord_scaler.transform(df_y_test[['LATITUDE', 'LONGITUDE']])

    for mlp_type in mlp_types:
        if mlp_type not in output_lens:
            output_lens[mlp_type] = len(df_train[mlp_col_dict[mlp_type]].unique())
            df_y_train.loc[:, mlp_col_dict[mlp_type]] = encoders[mlp_type].fit_transform(df_y_train[mlp_col_dict[mlp_type]].values.reshape(-1, 1)).reshape(-1,).astype(int)
            indexes[mlp_type] = [label_names.index(mlp_col_dict[mlp_type])]
            if mlp_type != 'cls-refloc':
                df_y_test.loc[:, mlp_col_dict[mlp_type]] = encoders[mlp_type].transform(df_y_test[mlp_col_dict[mlp_type]].values.reshape(-1, 1)).reshape(-1,).astype(int)

    y_train, y_test = df_y_train.values, df_y_test.values
    X_train, X_test = rssi_scaler.fit_transform(df_X_train.values), rssi_scaler.transform(df_X_test.values)

    return df_train, df_test, X_train, X_test, y_train, y_test, selected_aps, encoders, indexes, output_lens, num_aps, len(label_names) 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs for training')
    parser.add_argument('--min_std', default=5, type=int, help='Minimum standard deviation of RSSI distribution for each access point')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of cross-validation folds')
    parser.add_argument('--gnn', default='selection', type=str, help='Type of GNN backbone; either selection or attention')
    parser.add_argument('--ci', default='cls-refloc', type=str, help='How to generate coordinates (regression or classification): reg-coords or cls-refloc')
    parser.add_argument('--conf_mat', default=True, type=str2bool, help='Whether to generate a confusion matrix or not (True or False)')
    parser.add_argument('--plot_loss', default=True, type=str2bool, help='Whether to plot the learning curve after training')
    parser.add_argument('--param_search', default=False, type=str2bool, help='Whether to search for hyperparameters')
    args = vars(parser.parse_args())
    
    visualise = False
    find_params = args['param_search']
    conf_mat = args['conf_mat']
    plot_loss = args['plot_loss']
    gnn_type = args['gnn']
    min_std = args['min_std']
    epochs = args['epochs']
    num_folds = args['num_folds']
    mlp_types = ['cls-bld-flr', args['ci']]#, 'cls-bld', 'cls-flr', 'reg-coords'
    mlp_col_dict = {'cls-bld': 'BUILDINGID', 'cls-flr': 'FLOOR', 'cls-bld-flr': 'BLD_FLR', 'cls-refloc': 'REFLOC'}
    task_names = {
        'cls-bld': 'building classification',
        'cls-flr': 'floor classification',
        'cls-bld-flr': 'building-floor classification',
        'cls-refloc': 'reference location classification',
        'reg-coords': 'coordinate regression'
    }
    model_filename = gnn_type + '-gnn.pth'
    
    print('[INFO] Performing tasks:', ', '.join([task_names[mlp_type] for mlp_type in mlp_types]))

    print('[INFO] Loading the training dataset...')
    training_dataset_path = os.path.join('UJIndoorLoc', 'trainingData.csv')
    test_dataset_path = os.path.join('UJIndoorLoc', 'validationData.csv')
    df_train, df_test, X_train, X_test, y_train, y_test, selected_aps, encoders, indexes, output_lens, num_aps, output_size = load_data(
        training_dataset_path,
        test_dataset_path, 
        mlp_types, 
        mlp_col_dict, 
        min_std=min_std,
        visualise=visualise
    )
    
    if gnn_type == 'selection':
        best_params = {
            'gnn_type': 'selection',
            'batch_size': 128, 
            'learning_rate': 0.001, 
            'mlp-hidden': 32, 
            'num_filter_taps': 5, 
            'num_node_signals': 40, 
            'rssi_threshold': 5, 
            'weight_decay': 0.0001, 
            'pooling_size': 3, 
            'activation_func': torch.nn.ReLU, 
            'pooling_func': graphML.NoPool, 
            'k': 4, 
            'sigma': .5,
            'num_mlp_hidden': 0,
            'num_gnn_hidden': 2,
            'dropout': False
        }
    elif gnn_type == 'attention':
        best_params = {
            'gnn_type': 'attention',
            'activation_func': torch.nn.functional.relu, 
            'attention_heads': 1, 
            'batch_size': 64, 
            'learning_rate': 0.001, 
            'mlp-hidden': 32, 
            'num_filter_taps': 3, 
            'num_node_signals': 40, 
            'pooling_func': graphML.NoPool, 
            'pooling_size': 1, 
            'rssi_threshold': 10, 
            'weight_decay': 0.0001, 
            'k': 4, 
            'sigma': .5,
            'num_mlp_hidden': 0,
            'num_gnn_hidden': 2,
            'dropout': False
        }

    if find_params:
        print('[INFO] Starting parameter search...')
        best_params = find_best_params(
            X_train, 
            y_train, 
            selected_aps, 
            output_lens, 
            mlp_types, 
            indexes, 
            gnn_type, 
            num_aps, 
            output_size, 
            refloc_params=(df_train, df_test) if 'cls-refloc' in mlp_types else None, 
            epochs=20, 
            plot_loss=plot_loss, 
            params=best_params,
            k=num_folds
        )
        best_params['sigma'] = .5
        best_params['k'] = 4

    print('[INFO] Parameters:')
    print('\tGNN type:', gnn_type)
    print('\tActivation function:', 'ReLU' if str(best_params['activation_func']).lower().find('relu') >= 0 else 'tanh')
    print('\tPooling function:', 'no pooling' if str(best_params['pooling_func']).lower().find('nopool') >= 0 else 'max pooling')
    print('\tPooling size:', best_params['pooling_size'])
    print('\tLearning rate:', best_params['learning_rate'])
    print('\tBatch size:', best_params['batch_size'])
    print('\tWeight decay:', best_params['weight_decay'])
    print('\tEpochs:', epochs)
    print('\tWeight threshold (omega):', best_params['sigma'])
    print('\tMin standard deviation (sigma):', min_std)
    print('\tk:', best_params['k'])
    print('\tRSSI threshold:', best_params['rssi_threshold'])
    print('\tNumber of filters:', best_params['num_filter_taps'])
    print('\tDropout:', best_params['dropout'])
    print('\tNumber of GNN hidden layers:', best_params['num_gnn_hidden'])
    print('\tNumber of MLP hidden layers:', best_params['num_mlp_hidden'])
    print('\tNumber of GNN hidden units:', best_params['num_node_signals'])
    print('\tNumber of MLP hidden units:', best_params['mlp-hidden'])
    if gnn_type == 'attention':
        print('\tNumber of attention heads:', best_params['attention_heads'])

    print('[INFO] Started graph construction...')
    train_data, test_data = prepare_datasets(X_train, y_train, X_test, y_test, num_aps, output_size, reflocs='cls-refloc' in mlp_types, val=False)
    
    print('[INFO] Number of APs:', num_aps)
    Graph, W = generate_graph(X_train, selected_aps, rssi_threshold=best_params['rssi_threshold'])
    
    if not os.path.isfile(model_filename):
        print('[INFO] No local copy of the model found. Initiating the training process...')
        gnn_model = build_model(best_params, W, num_aps, output_lens, mlp_types)
        (gnn_model, loss_vals) = train_model(
            gnn_model, 
            train_data, 
            indexes=indexes,
            n_epochs=epochs, 
            learning_rate=best_params['learning_rate'], 
            batch_size=best_params['batch_size'], 
            weight_decay=best_params['weight_decay'],
            plot_loss=plot_loss
        )

        print('[INFO] Finished training and saved the model as', model_filename)
        torch.save(gnn_model, model_filename)
    else:
        print('[INFO] Loading a local copy of the model...')
        gnn_model = torch.load(model_filename)
    
    print('[INFO] Results:')
    print('Number of test samples:', y_test.shape[0])

    output = generate_predictions(
        gnn_model,
        test_data
    )
    
    results = evaluate(output, y_test, indexes, encoders, refloc_params=(df_train, df_test) if 'cls-refloc' in mlp_types else None, conf_mat=conf_mat, sigma=best_params['sigma'], k=best_params['k'], verbose=True)
            