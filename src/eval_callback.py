from mindspore import Callback
from mindspore import save_checkpoint


class EvalCallBack(Callback):
    """evaluation callback function"""
    def __init__(self, model0, eval_dataset, eval_per_epoch, epoch_per_eval0, config):
        self.model = model0
        self.eval_dataset = eval_dataset
        self.eval_per_epoch = eval_per_epoch
        self.epoch_per_eval = epoch_per_eval0
        self.best_acc = 0
        self.best_result = 0
        self.config = config

    def best_ckpt(self, epoch, accuracy, network):
        """save best checkpoint after eval during training"""
        if epoch // self.eval_per_epoch == 1:
            self.best_acc = accuracy
        elif epoch // self.eval_per_epoch > 1 and self.best_acc <= accuracy:
            self.best_acc = accuracy
            save_checkpoint(
                network,
                f"./{self.config.logs_dir}/best_acc.ckpt",
            )

    def epoch_end(self, run_context):
        """set evaluation at the end of each eval_per_epoch"""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epoch == 0:
            acc = self.model.eval(self.eval_dataset)
            self.best_ckpt(cur_epoch, acc["acc"], cb_param.network)
            self.epoch_per_eval["epoch"].append(cur_epoch)
            self.epoch_per_eval["acc"].append(acc)
            print(acc)