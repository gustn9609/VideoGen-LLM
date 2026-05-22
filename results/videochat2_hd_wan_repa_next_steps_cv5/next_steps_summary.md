# VideoChat2-HD Wan-REPA Next Steps Summary

| Group | Experiment | Acc | Correct/total |
|---|---|---:|---:|
| branch | branch_separated_eq_rel | 0.3542 | 34/96 |
| negative_control | negative_first_frame | 0.4688 | 45/96 |
| negative_control | negative_flow | 0.4688 | 45/96 |
| negative_control | negative_pixel | 0.4792 | 46/96 |
| negative_control | negative_random | 0.4792 | 46/96 |
| negative_control | negative_time_average | 0.4688 | 45/96 |
| schedule | schedule_constant_e3 | 0.4271 | 41/96 |
| schedule | schedule_cosine_decay_e3 | 0.4167 | 40/96 |
| schedule | schedule_late_start_e3 | 0.4167 | 40/96 |
| schedule | schedule_warmup_ce_polish_e3 | 0.4479 | 43/96 |

## Per-Type

### branch_separated_eq_rel
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.5417 | 13/24 |
| Motion-related Objects | 0.3333 | 8/24 |
| Repetition Count | 0.2083 | 5/24 |

### negative_first_frame
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.5833 | 14/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.3750 | 9/24 |

### negative_flow
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.2917 | 7/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.3750 | 9/24 |

### negative_pixel
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.3750 | 9/24 |

### negative_random
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.3750 | 9/24 |

### negative_time_average
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.2917 | 7/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.3750 | 9/24 |

### schedule_constant_e3
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5417 | 13/24 |
| Repetition Count | 0.2083 | 5/24 |

### schedule_cosine_decay_e3
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.1250 | 3/24 |

### schedule_late_start_e3
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3333 | 8/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.5833 | 14/24 |
| Repetition Count | 0.1250 | 3/24 |

### schedule_warmup_ce_polish_e3
| Type | Acc | Correct/total |
|---|---:|---:|
| Action Order | 0.3750 | 9/24 |
| Motion Recognition | 0.6250 | 15/24 |
| Motion-related Objects | 0.6667 | 16/24 |
| Repetition Count | 0.1250 | 3/24 |
