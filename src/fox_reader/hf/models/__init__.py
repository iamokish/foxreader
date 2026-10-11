"""Vendored model packages, one per ``model_type``.

Deliberately empty of imports: `fox_reader.hf.models.auto` resolves a model
type to its subpackage on demand, so loading the detector does not import the
recognisers' code. Re-exporting them here would undo that.
"""
