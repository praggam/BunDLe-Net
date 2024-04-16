import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import PCA
from functions import Database, preprocess_data, prep_data, BunDLeNet, train_model, plotting_neuronal_behavioural, plot_latent_timeseries, plot_phase_space, rotating_plot
import pickle

sys.path.append(r'../')

'''
### Load Data (excluding behavioural neurons) and plot
worm_num = 0
b_neurons = [
 	'AVAR',
 	'AVAL',
 	'SMDVR',
 	'SMDVL',
 	'SMDDR',
 	'SMDDL',
 	'RIBR',
 	'RIBL'
]
data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.states
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
plotting_neuronal_behavioural(X, B, state_names=state_names)
'''
#%% Prepare LETITGO Data

# Load data file
data_file = 'D:\LETITGO\LETITGO_001_MaZw\LETITGO_001_MaZw_ses_2_bundle-net.pickle'
try:
    file_to_read = open(data_file, "rb")
    data_dict = pickle.load(file_to_read)
    file_to_read.close()
except Exception as err:
    print(err)

# Define data variables
time_all = data_dict['time']
X_all = data_dict['X']
B_all = data_dict['B']

# Take only first run
i_start, i_stop = np.where(B_all == 1)[0][0], np.where(B_all == 2)[0][0]
B = B_all[i_start:i_stop]
time = time_all[i_start:i_stop]
X = X_all[i_start:i_stop,:]

# Remove 'start recording' state
i_cut = np.where(B == 1)[0][-1]+1
B = B[i_cut:]
time = time[i_cut:]
X = X[i_cut:,:]

# Apply new order to states
state_vars = np.unique(B)
for i_state in range(len(state_vars)):
    B[B == state_vars[i_state]] = i_state + 1
B = B.astype(np.int32)

state_names = ['close_cross', 'close_cue', 'close_plan', 'close_task_medium', 'close_rest', 'close_task_high', 'close_task_low',
               'open_cross', 'open_cue', 'open_plan', 'open_task_medium', 'open_rest', 'open_task_high', 'open_task_low']

plotting_neuronal_behavioural(X, B, state_names=state_names)


#%% Apply Bundle Net

### Preprocess and prepare data for BundLe Net
# time, X = preprocess_data(X, float(data.fps))
X_, B_ = prep_data(X, B, win=15)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

loss_array = train_model(
	X_,
	B_,
	model,
	optimizer,
	gamma=0.9, 
	n_epochs=2000,
	pca_init=False,
	best_of_5_init=False
)

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

algorithm = 'BunDLeNet'
### Save the weights (Uncomment to save and load for for later use)
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y0_, B_, state_names)
plot_phase_space(Y0_, B_, state_names = state_names)
rotating_plot(Y0_, B_,filename='figures/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

### Recurrence plot analysis of embedding
pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
plt.matshow(pd_Y, cmap='Greys')
plot_latent_timeseries(Y0_, B_, state_names)
plt.show()

