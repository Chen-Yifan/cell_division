from matplotlib import pyplot as plt

def visualize(result_path, img_data, real_data, predict_data, predicted_data, idx):
    
    f = plt.figure(figsize = (10,10))
    
    f.add_subplot(2,2,1)
    cs = plt.imshow(img_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('image')
    
    f.add_subplot(2,2,2)
    cs = plt.imshow(real_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('real')
    
    f.add_subplot(2,2,3)
    cs = plt.imshow(predict_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('predicted')
    
    f.add_subplot(2,2,4)
    cs = plt.imshow(predicted_data[idx,:,:])
    cbar = f.colorbar(cs)
    cbar.ax.minorticks_off()
    plt.title('pred_processed')	
    plt.show()
    plt.savefig(result_path+'/mask_comparison_idx%d.png'%(idx))
    plt.close()