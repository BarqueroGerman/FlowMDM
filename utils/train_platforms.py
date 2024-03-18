import os
import wandb

class TrainPlatform:
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name=None):
        pass

    def report_args(self, args, name):
        pass

    def close(self):
        pass


class WandbPlatform(TrainPlatform):
    def __init__(self, save_dir):
        path, name = os.path.split(save_dir)
        wandb.init(project='flowmdm',
                    name=name,
                    entity="TO_BE_FILLED", # fill with yours
                    )
        self.last_committed_iter = -1

    def report_scalar(self, name, value, iteration, group_name):
        wandb.log(data={name: value}, step=iteration, commit=True)#iteration != self.last_committed_iter)

    def report_data(self, data, iteration, group_name):
        # data = {name: value}
        wandb.log(data=data, step=iteration, commit=True)#iteration != self.last_committed_iter)
        self.last_committed_iter = iteration

    def report_args(self, args, name):
        wandb.config.update(args)

    def close(self):
        wandb.finish()


class ClearmlPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from clearml import Task
        path, name = os.path.split(save_dir)
        self.task = Task.init(project_name='motion_diffusion',
                              task_name=name,
                              output_uri=path)
        self.logger = self.task.get_logger()

    def report_scalar(self, name, value, iteration, group_name):
        self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_data(self, data, iteration, group_name):
        # data = {name: value}
        for name, value in data.items():
            self.logger.report_scalar(title=group_name, series=name, iteration=iteration, value=value)

    def report_args(self, args, name):
        self.task.connect(args, name=name)

    def close(self):
        self.task.close()


class TensorboardPlatform(TrainPlatform):
    def __init__(self, save_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=save_dir)

    def report_scalar(self, name, value, iteration, group_name):
        self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def report_data(self, data, iteration, group_name=None):
        # data = {name: value}
        for name, value in data.items():
            self.writer.add_scalar(f'{group_name}/{name}', value, iteration)

    def close(self):
        self.writer.close()


class NoPlatform(TrainPlatform):
    def __init__(self, save_dir):
        pass

    def report_scalar(self, name, value, iteration, group_name):
        pass
    
    def report_data(self, data, iteration, group_name=None):
        pass

    def close(self):
        pass

