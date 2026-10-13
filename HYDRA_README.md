The approach I took here with hydra is the early-validation approach using
ConfigStore pre-registration.  The following steps must be taken when writing any new
.yaml file


## Create a new dataclass

Define a new dataclass, inheriting it from any config group that it should be a part
of.  It will serve as a target for configurations.

For enum members that will be exposed to command-line overrides, use
`utils.MirroredEnum`.  This enables setting the enum value on the command-line.

```python
# in some file
@dataclass
class NewKindofDatasetOpts(DatasetOpts):
   ...
```

## Register the dataclass to a schema name

Add one line to `register_configs()` to register the schema name to the new
dataclass.  The rule I'm following is that each schema corresponds to exactly one
dataclass.  All overrides are provided as `.yaml` files.  Also, the name of the
schema should end with `_schema`. This ensures that the name won't conflict with the
name of a yaml file, which is also searched as a potential schema name (confusing).
For example:

```
# in opts.py
def register_configs() -> None:
   ...
   # here, 'data' will correspond to the conf directory opts/data
   cs.store(group="<some_config_group>", name="new_kind_of_dataset_schema", node=NewKindOfDatasetOpts)
```

## Create one or more new `.yaml` files

Create a `.yaml` file in some config group directory, for example:

`opts/<some_config_group>/my_override.yaml`

Inside the `.yaml` file, it will identify which schema it is targeting with:

```yaml
defaults:
  - new_kind_of_dataset_schema@_here_
  - _self_
```

## Main Program Setup

In your main program, you must call register_configs() and register_resolvers()
before calling main.  For example:

```python
# expt/train_general.py (or expt/test_expr_dataset.py)
if __name__ == "__main__":
	register_configs()
	register_resolvers()
	main()
```

And, in the main function you will have:

```python
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path="./opts", config_name="some_yaml_stem", version_base="1.2")
def main(cfg: DictConfig):
	opts: MyTopLevelOpts = OmegaConf.to_object(cfg)
    ...
```

where there is a file `opts/some_yaml_stem.yaml` with the content:

```yaml
defaults:
  - my_top_level_schema@_here_
  - _self_
  ...
```

and where `my_top_level_schema` is registered to `MyTopLevelOpts` inside
register_configs().

