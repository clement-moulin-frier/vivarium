from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AgentIdx(_message.Message):
    __slots__ = ["idx"]
    IDX_FIELD_NUMBER: _ClassVar[int]
    idx: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, idx: _Optional[_Iterable[int]] = ...) -> None: ...

class NDArray(_message.Message):
    __slots__ = ["ndarray"]
    NDARRAY_FIELD_NUMBER: _ClassVar[int]
    ndarray: bytes
    def __init__(self, ndarray: _Optional[bytes] = ...) -> None: ...

class RigidBody(_message.Message):
    __slots__ = ["center", "orientation"]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    center: NDArray
    orientation: NDArray
    def __init__(self, center: _Optional[_Union[NDArray, _Mapping]] = ..., orientation: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class SimulatorState(_message.Message):
    __slots__ = ["idx", "box_size", "n_agents", "n_objects", "num_steps_lax", "dt", "freq", "neighbor_radius", "to_jit", "use_fori_loop"]
    IDX_FIELD_NUMBER: _ClassVar[int]
    BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    N_OBJECTS_FIELD_NUMBER: _ClassVar[int]
    NUM_STEPS_LAX_FIELD_NUMBER: _ClassVar[int]
    DT_FIELD_NUMBER: _ClassVar[int]
    FREQ_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    TO_JIT_FIELD_NUMBER: _ClassVar[int]
    USE_FORI_LOOP_FIELD_NUMBER: _ClassVar[int]
    idx: NDArray
    box_size: NDArray
    n_agents: NDArray
    n_objects: NDArray
    num_steps_lax: NDArray
    dt: NDArray
    freq: NDArray
    neighbor_radius: NDArray
    to_jit: NDArray
    use_fori_loop: NDArray
    def __init__(self, idx: _Optional[_Union[NDArray, _Mapping]] = ..., box_size: _Optional[_Union[NDArray, _Mapping]] = ..., n_agents: _Optional[_Union[NDArray, _Mapping]] = ..., n_objects: _Optional[_Union[NDArray, _Mapping]] = ..., num_steps_lax: _Optional[_Union[NDArray, _Mapping]] = ..., dt: _Optional[_Union[NDArray, _Mapping]] = ..., freq: _Optional[_Union[NDArray, _Mapping]] = ..., neighbor_radius: _Optional[_Union[NDArray, _Mapping]] = ..., to_jit: _Optional[_Union[NDArray, _Mapping]] = ..., use_fori_loop: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class NVEState(_message.Message):
    __slots__ = ["position", "momentum", "force", "mass", "diameter", "entity_type", "entity_idx", "friction", "exists"]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    MOMENTUM_FIELD_NUMBER: _ClassVar[int]
    FORCE_FIELD_NUMBER: _ClassVar[int]
    MASS_FIELD_NUMBER: _ClassVar[int]
    DIAMETER_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_IDX_FIELD_NUMBER: _ClassVar[int]
    FRICTION_FIELD_NUMBER: _ClassVar[int]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    position: RigidBody
    momentum: RigidBody
    force: RigidBody
    mass: RigidBody
    diameter: NDArray
    entity_type: NDArray
    entity_idx: NDArray
    friction: NDArray
    exists: NDArray
    def __init__(self, position: _Optional[_Union[RigidBody, _Mapping]] = ..., momentum: _Optional[_Union[RigidBody, _Mapping]] = ..., force: _Optional[_Union[RigidBody, _Mapping]] = ..., mass: _Optional[_Union[RigidBody, _Mapping]] = ..., diameter: _Optional[_Union[NDArray, _Mapping]] = ..., entity_type: _Optional[_Union[NDArray, _Mapping]] = ..., entity_idx: _Optional[_Union[NDArray, _Mapping]] = ..., friction: _Optional[_Union[NDArray, _Mapping]] = ..., exists: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class AgentState(_message.Message):
    __slots__ = ["nve_idx", "prox", "motor", "behavior", "wheel_diameter", "speed_mul", "theta_mul", "proxs_dist_max", "proxs_cos_min", "color", "neighbor_map_dist", "neighbor_map_theta"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    PROX_FIELD_NUMBER: _ClassVar[int]
    MOTOR_FIELD_NUMBER: _ClassVar[int]
    BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
    WHEEL_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    SPEED_MUL_FIELD_NUMBER: _ClassVar[int]
    THETA_MUL_FIELD_NUMBER: _ClassVar[int]
    PROXS_DIST_MAX_FIELD_NUMBER: _ClassVar[int]
    PROXS_COS_MIN_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_MAP_DIST_FIELD_NUMBER: _ClassVar[int]
    NEIGHBOR_MAP_THETA_FIELD_NUMBER: _ClassVar[int]
    nve_idx: NDArray
    prox: NDArray
    motor: NDArray
    behavior: NDArray
    wheel_diameter: NDArray
    speed_mul: NDArray
    theta_mul: NDArray
    proxs_dist_max: NDArray
    proxs_cos_min: NDArray
    color: NDArray
    neighbor_map_dist: NDArray
    neighbor_map_theta: NDArray
    def __init__(self, nve_idx: _Optional[_Union[NDArray, _Mapping]] = ..., prox: _Optional[_Union[NDArray, _Mapping]] = ..., motor: _Optional[_Union[NDArray, _Mapping]] = ..., behavior: _Optional[_Union[NDArray, _Mapping]] = ..., wheel_diameter: _Optional[_Union[NDArray, _Mapping]] = ..., speed_mul: _Optional[_Union[NDArray, _Mapping]] = ..., theta_mul: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_dist_max: _Optional[_Union[NDArray, _Mapping]] = ..., proxs_cos_min: _Optional[_Union[NDArray, _Mapping]] = ..., color: _Optional[_Union[NDArray, _Mapping]] = ..., neighbor_map_dist: _Optional[_Union[NDArray, _Mapping]] = ..., neighbor_map_theta: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class ObjectState(_message.Message):
    __slots__ = ["nve_idx", "custom_field", "color"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_FIELD_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    nve_idx: NDArray
    custom_field: NDArray
    color: NDArray
    def __init__(self, nve_idx: _Optional[_Union[NDArray, _Mapping]] = ..., custom_field: _Optional[_Union[NDArray, _Mapping]] = ..., color: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class State(_message.Message):
    __slots__ = ["simulator_state", "nve_state", "agent_state", "object_state"]
    SIMULATOR_STATE_FIELD_NUMBER: _ClassVar[int]
    NVE_STATE_FIELD_NUMBER: _ClassVar[int]
    AGENT_STATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_STATE_FIELD_NUMBER: _ClassVar[int]
    simulator_state: SimulatorState
    nve_state: NVEState
    agent_state: AgentState
    object_state: ObjectState
    def __init__(self, simulator_state: _Optional[_Union[SimulatorState, _Mapping]] = ..., nve_state: _Optional[_Union[NVEState, _Mapping]] = ..., agent_state: _Optional[_Union[AgentState, _Mapping]] = ..., object_state: _Optional[_Union[ObjectState, _Mapping]] = ...) -> None: ...

class StateChange(_message.Message):
    __slots__ = ["nve_idx", "col_idx", "nested_field", "value"]
    NVE_IDX_FIELD_NUMBER: _ClassVar[int]
    COL_IDX_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELD_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    nve_idx: _containers.RepeatedScalarFieldContainer[int]
    col_idx: _containers.RepeatedScalarFieldContainer[int]
    nested_field: _containers.RepeatedScalarFieldContainer[str]
    value: NDArray
    def __init__(self, nve_idx: _Optional[_Iterable[int]] = ..., col_idx: _Optional[_Iterable[int]] = ..., nested_field: _Optional[_Iterable[str]] = ..., value: _Optional[_Union[NDArray, _Mapping]] = ...) -> None: ...

class AddAgentInput(_message.Message):
    __slots__ = ["n_agents", "serialized_config"]
    N_AGENTS_FIELD_NUMBER: _ClassVar[int]
    SERIALIZED_CONFIG_FIELD_NUMBER: _ClassVar[int]
    n_agents: int
    serialized_config: str
    def __init__(self, n_agents: _Optional[int] = ..., serialized_config: _Optional[str] = ...) -> None: ...

class IsStartedState(_message.Message):
    __slots__ = ["is_started"]
    IS_STARTED_FIELD_NUMBER: _ClassVar[int]
    is_started: bool
    def __init__(self, is_started: bool = ...) -> None: ...
