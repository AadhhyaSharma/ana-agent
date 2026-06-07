#!/usr/bin/env python3
"""
core_infra.py
=============
Core Infrastructure Layer for ANA Agent.
Covers: Config, Logging, EventBus, Scheduler, Metrics,
        ServiceRegistry, PluginSystem, HealthMonitor,
        Cache, RateLimiter, CircuitBreaker, IPC Server.
"""

import os, sys, json, time, math, random, hashlib, threading, queue
import datetime, traceback, socket, struct, re, functools, itertools
from collections import defaultdict, deque, OrderedDict
from typing import List, Dict, Tuple, Optional, Any, Callable, Set
from enum import Enum, auto

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class CV:
    """A single configuration value with type, validation, mutability, history."""
    TYPES = {"str":str,"int":int,"float":float,"bool":bool,"list":list,"dict":dict}

    def __init__(self, key, default, tname="str", desc="", validator=None, mutable=True, secret=False):
        self.key=key; self.default=default; self.tname=tname; self.desc=desc
        self.validator=validator; self.mutable=mutable; self.secret=secret
        self._v=default; self._changed=False; self._hist=deque(maxlen=20)

    @property
    def value(self): return self._v

    @value.setter
    def value(self, v):
        if not self.mutable: raise PermissionError(f"'{self.key}' immutable")
        if self.validator and not self.validator(v): raise ValueError(f"Validation failed: {self.key}={v}")
        old=self._v; self._v=v; self._changed=True
        self._hist.append({"from":old,"to":v,"at":time.time()})

    def reset(self): self._v=self.default; self._changed=False
    def to_dict(self):
        return {"key":self.key,"value":"***" if self.secret else self._v,
                "default":self.default,"tname":self.tname,"changed":self._changed}
    def __repr__(self): return f"CV({self.key}={'***' if self.secret else self._v})"


class Config:
    PRE = "ANA_"
    SCHEMA = {
        "agent.name":("ANA","str","Agent name"),
        "agent.version":("2.0.0","str","Version",False),
        "agent.debug":(False,"bool","Debug mode"),
        "agent.log_level":("INFO","str","Log level"),
        "brain.lr":(0.001,"float","Learning rate"),
        "brain.warmup":(100,"int","LR warmup"),
        "brain.decay":(0.99,"float","LR decay"),
        "mem.cap":(1024,"int","Memory capacity"),
        "mem.consolidate":(5.0,"float","Consolidation threshold"),
        "learn.batch":(16,"int","Batch size"),
        "persist.path":("ana_state.json","str","Save path"),
        "persist.backup":(True,"bool","Backup on save"),
        "log.file":("ana.log","str","Log file"),
        "log.maxmb":(10,"int","Max log MB"),
        "metrics.on":(True,"bool","Enable metrics"),
    }

    def __init__(self):
        self._vals: Dict[str,CV]={}; self._watchers: Dict[str,List[Callable]]=defaultdict(list)
        self._files: List[str]=[]; self._load(); self._env()

    def _load(self):
        for key,spec in self.SCHEMA.items():
            if len(spec)==3: default,tname,desc=spec; mut=True
            else: default,tname,desc,mut=spec
            self._vals[key]=CV(key,default,tname,desc,mutable=mut)

    def _env(self):
        for key,cv in self._vals.items():
            ekey=self.PRE+key.upper().replace(".","_")
            if ekey in os.environ:
                raw=os.environ[ekey]
                try:
                    if cv.tname=="bool": cv.value=raw.lower() in("1","true","yes")
                    elif cv.tname in("list","dict"): cv.value=json.loads(raw)
                    else: cv.value=CV.TYPES.get(cv.tname,str)(raw)
                except Exception: pass

    def get(self, key, default=None): return self._vals[key].value if key in self._vals else default
    def set(self, key, value):
        if key in self._vals:
            old=self._vals[key].value; self._vals[key].value=value
            for w in self._watchers.get(key,[]):
                try: w(key,old,value)
                except Exception: pass
        else:
            cv=CV(key,value); cv._v=value; self._vals[key]=cv
    def watch(self, key, fn): self._watchers[key].append(fn)
    def load_file(self, path):
        if not os.path.exists(path): return False
        try:
            with open(path) as f: data=json.load(f)
            for k,v in data.items(): self.set(k,v)
            self._files.append(path); return True
        except Exception: return False
    def save_file(self, path):
        try:
            with open(path,"w") as f: json.dump({k:v.value for k,v in self._vals.items() if not v.secret},f,indent=2)
            return True
        except Exception: return False
    def section(self, prefix): return {k:v.value for k,v in self._vals.items() if k.startswith(prefix+".")}
    def diff(self): return {k:{"default":v.default,"current":v.value} for k,v in self._vals.items() if v._changed}
    def all(self): return {k:v.to_dict() for k,v in self._vals.items()}
    def __repr__(self): return f"Config({len(self._vals)} keys)"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class LL(Enum):
    DEBUG=10; INFO=20; WARNING=30; ERROR=40; CRITICAL=50


class LR:
    def __init__(self, level, msg, src="", extra=None, exc=None):
        self.level=level; self.msg=msg; self.src=src
        self.extra=extra or {}; self.exc=exc
        self.ts=time.time(); self.dt=datetime.datetime.now().isoformat()
        self.tid=threading.get_ident()
    def fmt(self, color=False):
        C={LL.DEBUG:"\033[36m",LL.INFO:"\033[32m",LL.WARNING:"\033[33m",
           LL.ERROR:"\033[31m",LL.CRITICAL:"\033[35m"}
        R="\033[0m"; ls=f"[{self.level.name:8s}]"
        if color: ls=C.get(self.level,"")+ls+R
        return f"{self.dt} {ls}{' ('+self.src+')' if self.src else ''} {self.msg}"
    def to_dict(self): return {"ts":self.dt,"level":self.level.name,"msg":self.msg,"src":self.src}


class ConH:
    def __init__(self, min_l=LL.INFO, color=True): self.ml=min_l; self.color=color
    def handle(self, r):
        if r.level.value>=self.ml.value: print(r.fmt(self.color))


class FileH:
    def __init__(self, path, min_l=LL.DEBUG, maxb=10_000_000, bk=3):
        self.path=path; self.ml=min_l; self.maxb=maxb; self.bk=bk; self._lk=threading.Lock()
    def handle(self, r):
        if r.level.value<self.ml.value: return
        with self._lk:
            try:
                if os.path.exists(self.path) and os.path.getsize(self.path)>self.maxb:
                    for i in range(self.bk-1,0,-1):
                        s=f"{self.path}.{i}"; d=f"{self.path}.{i+1}"
                        if os.path.exists(s): os.replace(s,d)
                    if os.path.exists(self.path): os.replace(self.path,f"{self.path}.1")
                with open(self.path,"a",encoding="utf-8") as f: f.write(r.fmt(False)+"\n")
            except Exception: pass


class BufH:
    def __init__(self, cap=1000, min_l=LL.DEBUG): self.ml=min_l; self.recs=deque(maxlen=cap)
    def handle(self, r):
        if r.level.value>=self.ml.value: self.recs.append(r)
    def recent(self, n=50): return list(self.recs)[-n:]


class ALog:
    def __init__(self, name="ana"):
        self.name=name; self.handlers=[]; self._lk=threading.Lock(); self._cnt=defaultdict(int)
        self.buf=BufH(1000); self.add(self.buf)
    def add(self, h): self.handlers.append(h)
    def _log(self, level, msg, src="", extra=None, exc=None):
        r=LR(level,msg,src or self.name,extra,exc); self._cnt[level.name]+=1
        with self._lk:
            for h in self.handlers:
                try: h.handle(r)
                except Exception: pass
    def debug(self, m, s=""): self._log(LL.DEBUG,m,s)
    def info(self, m, s=""): self._log(LL.INFO,m,s)
    def warning(self, m, s=""): self._log(LL.WARNING,m,s)
    def error(self, m, s=""): self._log(LL.ERROR,m,s)
    def critical(self, m, s=""): self._log(LL.CRITICAL,m,s)
    def exc(self, m, s=""): self._log(LL.ERROR,m,s,exc=traceback.format_exc())
    def recent(self, n=20): return [r.fmt() for r in self.buf.recent(n)]
    def stats(self): return {"cnt":dict(self._cnt),"handlers":len(self.handlers)}


log = ALog("ana")
log.add(ConH(LL.WARNING))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — EVENT BUS
# ═══════════════════════════════════════════════════════════════════════════════

class EP(Enum):
    LOW=0; NORMAL=1; HIGH=2; URGENT=3


class Evt:
    _ctr=itertools.count(0)
    def __init__(self, etype, data=None, src="", pri=EP.NORMAL):
        self.id=next(Evt._ctr); self.etype=etype; self.data=data
        self.src=src; self.pri=pri; self.ts=time.time()
        self.handled=False; self.cancelled=False; self.meta={}
    def cancel(self): self.cancelled=True
    def to_dict(self): return {"id":self.id,"type":self.etype,"src":self.src,"ts":self.ts}
    def __repr__(self): return f"Evt({self.id}:{self.etype})"


class Bus:
    def __init__(self, maxhist=10000):
        self._subs: Dict[str,List[Tuple[Callable,int]]]=defaultdict(list)
        self._wcs: List[Tuple[str,Callable,int]]=[]
        self._hist=deque(maxlen=maxhist); self._stats=defaultdict(int)
        self._lk=threading.RLock(); self._mw: List[Callable]=[]
        self._dead=deque(maxlen=100)

    def sub(self, etype, fn, pri=0):
        with self._lk:
            if "*" in etype: self._wcs.append((etype,fn,pri))
            else:
                self._subs[etype].append((fn,pri))
                self._subs[etype].sort(key=lambda x:-x[1])
        return hashlib.md5(f"{etype}{id(fn)}{time.time()}".encode()).hexdigest()[:8]

    def unsub(self, etype, fn):
        with self._lk:
            if etype in self._subs:
                self._subs[etype]=[(h,p) for h,p in self._subs[etype] if h!=fn]

    def add_mw(self, fn): self._mw.append(fn)

    def pub(self, evt):
        if evt.cancelled: return 0
        for mw in self._mw:
            try:
                evt=mw(evt)
                if evt is None or evt.cancelled: return 0
            except Exception: pass
        self._hist.append(evt); self._stats[evt.etype]+=1; self._stats["_t"]+=1
        with self._lk:
            direct=list(self._subs.get(evt.etype,[]))
            wcs=[(p,h) for pat,h,p in self._wcs if self._match(pat,evt.etype)]
        all_h=[(h,p) for h,p in direct]+[(h,p) for p,h in wcs]
        all_h.sort(key=lambda x:-x[1]); n=0
        for fn,_ in all_h:
            if evt.cancelled: break
            try: fn(evt); evt.handled=True; n+=1
            except Exception as e:
                log.error(f"Handler err:{e}","Bus")
                self._dead.append({"evt":evt.to_dict(),"err":str(e)})
        if not evt.handled: self._dead.append({"evt":evt.to_dict(),"reason":"no_handlers"})
        return n

    def emit(self, etype, data=None, src="", pri=EP.NORMAL):
        return self.pub(Evt(etype,data,src,pri))

    def _match(self, pat, etype):
        pp=pat.split("."); pe=etype.split(".")
        if len(pp)!=len(pe) and "**" not in pat: return False
        for a,b in zip(pp,pe):
            if a=="**": return True
            if a!="*" and a!=b: return False
        return True

    def history(self, etype=None, n=50):
        if etype: return [e for e in self._hist if e.etype==etype][-n:]
        return list(self._hist)[-n:]

    def stats(self): return {"total":self._stats["_t"],"by_type":{k:v for k,v in self._stats.items() if k!="_t"},"dead":len(self._dead)}
    def __repr__(self): return f"Bus(subs={len(self._subs)},pub={self._stats['_t']})"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METRICS & HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

class Metrics:
    """System metrics and health monitoring."""
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.start_time = time.time()

    def inc(self, name, val=1):
        self.counters[name] += val

    def set(self, name, val):
        self.gauges[name] = val

    def observe(self, name, val):
        self.histograms[name].append(val)
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-500:]

    def uptime(self):
        return time.time() - self.start_time

    def to_dict(self):
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "uptime": self.uptime(),
            "timestamp": time.time()
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — INFRASTRUCTURE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class Infra:
    """Central infrastructure manager."""
    def __init__(self):
        self.config = Config()
        self.log = log
        self.bus = Bus()
        self.metrics = Metrics()
        self._started = False

    def start(self):
        """Initialize infrastructure."""
        if self._started:
            return
        
        self.log.info("Initializing ANA Infrastructure...")
        
        # Setup logging
        log_file = self.config.get("log.file", "ana.log")
        if log_file:
            self.log.add(FileH(log_file, LL.DEBUG))
        
        self.log.info(f"ANA v{self.config.get('agent.version')}")
        self.log.info(f"Config loaded: {len(self.config._vals)} keys")
        self.log.info(f"Event bus initialized")
        self.log.info(f"Metrics enabled: {self.config.get('metrics.on')}")
        
        self._started = True
        self.bus.emit("infra.started", {"timestamp": time.time()})

    def stop(self):
        """Shutdown infrastructure."""
        if not self._started:
            return
        
        self.log.info("Shutting down ANA Infrastructure...")
        self.bus.emit("infra.stopping", {"timestamp": time.time()})
        self._started = False

    def health(self):
        """Get system health status."""
        return {
            "started": self._started,
            "uptime": self.metrics.uptime(),
            "log_stats": self.log.stats(),
            "bus_stats": self.bus.stats(),
            "metrics": self.metrics.to_dict()
        }

    def __repr__(self):
        return f"Infra(started={self._started}, config={len(self.config._vals)})"


# Global infrastructure instance
infra = Infra()
