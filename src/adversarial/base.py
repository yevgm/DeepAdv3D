from typing import Tuple

import torch
import tqdm

from utils import laplacebeltrami_FEM_v2
from utils.misc import check_data

class AdversarialExample(object):
  def __init__(self,
      pos:torch.Tensor,
      edges:torch.LongTensor,
      faces:torch.LongTensor,
      classifier:torch.nn.Module,
      classifier_args:dict,
      target:int=None):
    super().__init__()
    check_data(pos, edges, faces, float_type=torch.float)
    float_type = pos.dtype

    self._pos = pos
    self._faces = faces
    self._edges = edges
    self._classifier = classifier
    self._target = None if target is None else torch.tensor([target], device=pos.device, dtype=torch.long)
    self._classifier_args = classifier_args

    # compute useful data
    self._stiff, self._area = laplacebeltrami_FEM_v2(self._pos, self._faces)

  @property
  def pos(self)->torch.Tensor: return self._pos
  @property
  def edges(self)->torch.LongTensor:return self._edges
  @property
  def faces(self)->torch.LongTensor:return self._faces
  @property
  def target(self)->torch.Tensor:return self._target
  @property
  def stiff(self)->Tuple: return self._stiff
  @property
  def area(self)->Tuple: return self._area
  @property
  def classifier(self)->torch.nn.Module: return self._classifier
  @property
  def vertex_count(self)->int: return self._pos.shape[0]
  @property
  def edge_count(self)->int: return self._edges.shape[0]
  @property
  def face_count(self)->int: return self._faces.shape[0]

  @property
  def device(self) ->torch.device:  return self.pos.device
  @property
  def dtype_int(self)->torch.dtype: return self.edges.dtype
  @property
  def dtype_float(self)->torch.dtype: return self.pos.dtype

  @property
  def perturbed_pos(self) -> torch.Tensor:
    raise NotImplementedError()

  @property
  def logits(self) -> torch.Tensor:
    Z,_,_ = self.classifier(self.pos, **self._classifier_args)
    return Z

  @property
  def perturbed_logits(self)->torch.Tensor:
    pZ,_,_ = self.classifier(self.perturbed_pos, **self._classifier_args)
    return pZ

  @property
  def is_targeted(self)->bool: return self._target is not None

  # cached operations
  @property
  def is_successful(self) -> bool:
    prediction = self.logits.argmax().item()
    adversarial_prediction = self.perturbed_logits.argmax().item()

    if self.is_targeted:
      return adversarial_prediction == self.target
    else:
      return  prediction != adversarial_prediction

class Builder(object):
  def __init__(self):
    super().__init__()
    self.adex_data = dict()

  def set_mesh(self, pos, edges, faces):
    self.adex_data["pos"] = pos
    self.adex_data["edges"] = edges
    self.adex_data["faces"] = faces
    return self
    
  def set_target(self, t:int):
    self.adex_data["target"] = t
    return self

  def set_classifier(self, classifier:torch.nn.Module):
    self.adex_data["classifier"] = classifier
    return self

  def build(self, **args)->AdversarialExample: #the dictionary args contains additional parameters
    raise NotImplementedError()                #whose meaning will depend on the sub-class

class LossFunction(object):
    def __init__(self, adv_example:AdversarialExample):
        self.adv_example = adv_example

    def __call__(self) -> torch.Tensor:
        raise NotImplementedError