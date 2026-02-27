
import gym
import gym_super_mario_bros
import os
import torch
import argparse
import wandb

from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from EnvProperties.f_env import envWrapper, envPreproccessing, getEnvinfo, foveatedObservation, SkipFrame
from Utils.Utils import seeding, loadConfig, ensureParentFile, saveLoss, plotMetrics
# from EnvProperties.f_env import envWrapper, envPreproccessing, getEnvinfo, foveatedObservation
# from EnvProperties.env import
from DreamerCore.CoreInit import CentralInitialization
from DreamerCore.WorldModel import WorldModel
from DreamerCore.BehaviourModel import BehaviorModel
from DreamerCore.EnvInteraction import EnvironmentInteraction
from DreamerCore.SaveCheckPoints import Savingcheckpoints
from DreamerCore.LoadCheckPoints import LoadCheckpoints

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(configFile):
  config = loadConfig(configFile)
  seeding(config.seed)

  wandb.init(
      project="neuro-dreamer-mario",
      name=config.run_name,
      config=config
  )

  runName = f'{config.envname}_{config.run_name}'
  checkpointLoad = os.path.join(config.folderName.checkpointFolder, f'{runName}_{config.checkpointLoad}')
  metricsFilename = os.path.join(config.folderName.metricFolder, runName)
  plotFilename = os.path.join(config.folderName.plotsfolder, runName)
  checkpointFilenameBase = os.path.join(config.folderName.checkpointFolder, runName)
  videoFilenameBase = os.path.join(config.folderName.videoFolder, runName)

  ensureParentFile(metricsFilename, plotFilename, checkpointFilenameBase, videoFilenameBase)
  base_env = gym_super_mario_bros.make(config.envname)
  base_env = JoypadSpace(base_env, RIGHT_ONLY)
  base_env = SkipFrame(base_env, skip=4)
  
  env = envWrapper(envPreproccessing(
      gym.wrappers.ResizeObservation(base_env, (64, 64))))
  if config.use_foveation:
    base_env = foveatedObservation(base_env)



  base_enveval = gym_super_mario_bros.make(config.envname)
  base_enveval = JoypadSpace(base_enveval, RIGHT_ONLY)
  base_enveval = SkipFrame(base_enveval, skip=4)

  enveval = envWrapper(envPreproccessing(
      gym.wrappers.ResizeObservation(base_enveval, (64, 64))))
  if config.use_foveation:
    base_enveval = foveatedObservation(base_enveval)



  observation_shape , action_size = getEnvinfo(env)

  print(f'Observation shape: {observation_shape}', f'Action size: {action_size}')

  core = CentralInitialization(observation_shape, action_size, device, config)
  worldmodel = WorldModel(core, config)
  behaviour = BehaviorModel(core, config)
  envinter = EnvironmentInteraction(core, config)
  loadCheckpoints = LoadCheckpoints(core, worldmodel, behaviour, envinter, config)
  if config.resume:
    loadCheckpoints.loadCheckPoint(checkpointLoad)
  saveCheckpoints = Savingcheckpoints(core, worldmodel, behaviour, envinter, config)

  envinter.envInteraction(env, config.episodedbeforestart, seed=config.seed)

  iterationsNum = config.gradientSteps // config.replayRatio
  for _ in range(iterationsNum):
    for _ in range(config.replayRatio):
      sampleData = core.buffer.sample(config.dreamer.batchsize, config.dreamer.batchlength)
      initialStates, worldModelMetrics = worldmodel.train_world(sampleData)
      behaviourMetrics = behaviour.train_behaviour(initialStates)
      core.totalGradientSteps += 1

      if core.totalGradientSteps % 100 == 0:
        wandb.log({
            "gradientSteps": core.totalGradientSteps,
            **worldModelMetrics,
            **behaviourMetrics
        })

      if core.totalGradientSteps % config.checkpointInterval == 0 and config.saveCheckpoints:
        print("Steps:", core.totalGradientSteps,
          "ReconLoss:", worldModelMetrics["reconstructionLoss"],
          "Reward:", worldModelMetrics.get("rewardLoss", 0))
        suffix = f'{core.totalGradientSteps/1000:.0f}k'
        saveCheckpoints.saveCheckpoint(f'{checkpointFilenameBase}_{suffix}', core.totalGradientSteps)
        evalScore = envinter.envInteraction(enveval, config.numEvaluationEpisode, seed=config.seed, evaluation=True, savevideo=True, fileName=f'{videoFilenameBase}_{suffix}')
        print(f'Saved Checkpoint and Video to {suffix:>6} gradient steps. Evaluation Score: {evalScore:>0.2f}')

    RecentScore = envinter.envInteraction(env, config.numInteractionEpisode, seed=config.seed)
    if config.saveMetrics:
      metricBase = {'envSteps': core.totalEnvSteps,
                    'gradientSteps': core.totalGradientSteps,
                    'totalReward': RecentScore }
      combinedMetrics = metricBase | worldModelMetrics | behaviourMetrics

      saveLoss(metricsFilename, metricBase | worldModelMetrics | behaviourMetrics)
      # plotMetrics(f'{metricsFilename}', savePath=f'{plotFilename}', title=f'{config.envname}')
      wandb.log(combinedMetrics)
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default='Config/config.yml')
  main(parser.parse_args().config)
