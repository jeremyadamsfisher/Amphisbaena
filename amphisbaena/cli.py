import os
from functools import partial

import click
import yaml
from loguru import logger
from pytorch_lightning.loggers import WandbLogger

import wandb
from amphisbaena.train import train as train_


def save_artifact(run, res):
    artifact = wandb.Artifact(name="amphisbaena", type="model")
    artifact.add_file(local_path=os.path.join(run.dir, res.checkpoint_most_accurate))
    run.log_artifact(artifact)


def sweep_iteration(max_epochs, save_most_accurate):
    with wandb.init() as run:
        res = train_(
            lr=wandb.config.lr,
            backbone=wandb.config.backbone,
            max_epochs=max_epochs,
            save_most_accurate=save_most_accurate,
            trainer_config={"logger": WandbLogger()},
        )
        if save_most_accurate:
            save_artifact(run, res)


@click.group()
@click.option("--wandb-project", type=str)
@click.option("--log-model/--no-log-model", default=True, type=bool)
@click.pass_context
def cli(ctx, wandb_project, log_model):
    ctx.ensure_object(dict)
    ctx.obj["wandb_project"] = wandb_project
    ctx.obj["save_most_accurate"] = log_model


@cli.command()
@click.option("--lr", default=1e-2, type=float, help="Learning rate")
@click.option("--max-epochs", type=int, default=10)
@click.option(
    "--backbone",
    default="lin",
    help="image recognition backbone type",
    type=click.Choice(["lin", "conv", "conv_pretrained"], case_sensitive=False),
)
@click.pass_context
def train(ctx, lr, backbone, max_epochs):
    """Train an Amphisbaena"""
    with wandb.init(project=ctx.obj["wandb_project"]) as run:
        res = train_(
            lr=lr,
            backbone=backbone,
            max_epochs=max_epochs,
            save_most_accurate=ctx.obj["save_most_accurate"],
            trainer_config={"logger": WandbLogger()},
        )
        if ctx.obj["save_most_accurate"]:
            save_artifact(run, res)


@cli.command()
@click.option("--sweep-id", type=str)
@click.option("--count", type=int, default=10)
@click.option("--max-epochs", type=int, default=10)
@click.option("--sweep-config-fp", type=click.Path())
@click.pass_context
def sweep(ctx, sweep_id, count, max_epochs, sweep_config_fp):
    """Explore hyperparameter space"""
    if sweep_id and sweep_config_fp or (not sweep_id and not sweep_config_fp):
        raise Exception(
            "Please supply either an existing sweep id or specify the "
            "configuration file (yaml) to specify a new sweep"
        )
    wandb_project = ctx.obj["wandb_project"]
    if sweep_id is None:
        with open(sweep_config_fp) as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project=wandb_project)
        logger.info("new sweep created: {}", sweep_id)
        logger.info(
            "continue sweep later with: `amphisbaena --wandb-project {} sweep --max-epochs {} --sweep-id {}`",
            wandb_project,
            max_epochs,
            sweep_id,
        )
    wandb.agent(
        sweep_id,
        function=partial(
            sweep_iteration, max_epochs=max_epochs, save_most_accurate=True
        ),
        count=count,
        project=wandb_project,
    )


if __name__ == "__main__":
    cli(obj={})
