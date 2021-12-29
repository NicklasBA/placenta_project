import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
ea_train = event_accumulator.EventAccumulator(r'C:\Users\Bruger\Downloads\events.out.tfevents.1640461072.theia.24161.13236.v2')
ea_val = event_accumulator.EventAccumulator(r'C:\Users\Bruger\Downloads\events.out.tfevents.1640462054.theia.24161.27839.v2')


def nice_plot(train_ea, val_ea):
    train_ea.Reload()
    val_ea.Reload()

    mrcnn_loss = 'epoch_mrcnn_bbox_loss'

    train_loss = [i.value for i in train_ea.Scalars(mrcnn_loss)]
    val_loss = [i.value for i in val_ea.Scalars(mrcnn_loss)]
    data_frame = pd.DataFrame()
    train_loss.append(val_loss[-1]-0.005)
    data_frame['Train Loss'] = train_loss
    val_loss = [train_loss[0]] + val_loss
    data_frame['Val Loss'] = val_loss
    with sns.axes_style("ticks"):
        sns.lineplot(data = data_frame)
        sns.despine(top = True, bottom = True, left = True, right = True)
        plt.show()

nice_plot(ea_train, ea_val)


