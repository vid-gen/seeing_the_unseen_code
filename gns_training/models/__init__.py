# The upstream htung0101/Physion-particles@5ac002a models/__init__.py eagerly imported
# DPINet, DPINet2, GNS, GNSRigid and GNSRigidH (which pull in torchvision / ipdb).
# Those legacy models are not part of this release, so the eager imports are removed;
# use `from models.GNS_pyg import ...` directly.
