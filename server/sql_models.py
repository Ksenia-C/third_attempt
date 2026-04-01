from datetime import datetime
from models import *

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Select, create_engine, desc, func,true,  Column, Integer, String, Boolean, Text, Float, ForeignKey, Table, Date, Enum
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy.orm import Session

Base = declarative_base()

class SimulationRuns(Base):
    __tablename__ = 'simulation_runs'

    run_id: Mapped[str] = mapped_column(primary_key=True)
    fl_algorithm: Mapped[FLAlgorithm] = mapped_column(Enum(FLAlgorithm))
    distribution: Mapped[Distribution] = mapped_column(Enum(Distribution))
    distribution_params: Mapped[str] = mapped_column(Text)
    run_time: Mapped[datetime] = mapped_column(Date, default=datetime.now())  
    
    progress: Mapped['RunProgress'] = relationship(back_populates="related_run", uselist=False, cascade='all, delete-orphan')
    stats: Mapped['RunStats'] = relationship(back_populates="related_run", uselist=False, cascade='all, delete-orphan')


class RunProgress(Base):
    __tablename__ = 'run_progress'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(ForeignKey('simulation_runs.run_id'), unique=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus))
    result: Mapped[str | None] = mapped_column(Text)

    related_run: Mapped["SimulationRuns"] = relationship(back_populates="progress")

class RunStats(Base):
    __tablename__ = 'run_stats'
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    run_id: Mapped[str] = mapped_column(ForeignKey('simulation_runs.run_id'), unique=True)
    simulation_time_ms: Mapped[float | None] = mapped_column(Integer)
    run_time: Mapped[datetime] = mapped_column(Date, default=datetime.now())  

    related_run: Mapped["SimulationRuns"] = relationship(back_populates="stats")
