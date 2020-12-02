import glob
import os
import json

import numpy as np
import pandas as pd
import itertools

import util

SIDS = range(1, 51)
MODELS = ['onestat', 'twostat', 'propwaves']
NOISES = [0, 1]
NSIMS_PER_SUBJECT = 300
PREPSIM_SNR = 100.

SIM_BATCH = 8

localrules: aggreg_sim, aggreg_rec, plot_rec_taas, md2pdf

rule preparatory_simulate:
    # To estimate the background noise level
    input:
        surf="data/Geometry/id001/surface.npz",
        seeg="data/Geometry/id001/seeg.txt",
    output:
        results="run/Taa/prep/results_{model}_{noise}.pkl"
    resources:
        mem_mb=4096
    run:
        util.simulate("id001", input.surf, input.seeg, wildcards.model, PREPSIM_SNR, int(wildcards.noise),
                      60, None, output.results, None)


rule mesh_dependence:
    input:
        surf0="data/Geometry/id001/surface.npz",
        surf1="data/GeometryFine/id001/surface.npz",
        seeg="data/Geometry/id001/seeg.txt",
        prepres="run/Taa/prep/results_propwaves_0.pkl",
        recres="run/Taa/df-results-rec.pkl"
    output:
        configs="run/Taa/meshdep/configs.pkl",
        results="run/Taa/meshdep/results.pkl",
        grouped="run/Taa/meshdep/grouped.pkl"
    resources:
        mem_mb=4096
    run:
        dfs = pd.read_pickle(input.prepres)
        dfr = pd.read_pickle(input.recres)
        dfs = dfs[dfs.detconf == 0]
        dfr = dfr[dfr.detconf == 0]
        snr = PREPSIM_SNR * 10**(np.percentile(dfr.lnta_p80, 95) - np.percentile(dfs.lnta_p80, 95))
        util.mesh_dependence("id001", input.surf0, input.surf1, input.seeg, snr, 0, 60,
                             output.configs, output.results, output.grouped)


rule simulate:
    input:
        surf="data/Geometry/{subject}/surface.npz",
        seeg="data/Geometry/{subject}/seeg.txt",
        prepres="run/Taa/prep/results_{model}_{noise}.pkl",
        recres="run/Taa/df-results-rec.pkl"
    output:
        configs="run/Taa/simulations/configs_{subject}_{model}_{noise}.pkl",
        results="run/Taa/simulations/results_{subject}_{model}_{noise}.pkl",
        grouped="run/Taa/simulations/grouped_{subject}_{model}_{noise}.pkl",
    resources:
        mem_mb=8192
    group: "simgroup"
    run:
        dfs = pd.read_pickle(input.prepres)
        dfr = pd.read_pickle(input.recres)
        dfs = dfs[dfs.detconf == 0]
        dfr = dfr[dfr.detconf == 0]
        snr = PREPSIM_SNR * 10**(np.percentile(dfr.lnta_p80, 95) - np.percentile(dfs.lnta_p80, 95))
        util.simulate(wildcards.subject, input.surf, input.seeg, wildcards.model, snr, int(wildcards.noise),
                      NSIMS_PER_SUBJECT, output.configs, output.results, output.grouped, plot='none')


# Batching the simulation tasks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

sims = list(itertools.product(SIDS, MODELS, NOISES))
for i, chunk in enumerate(chunks(sims, SIM_BATCH)):
    rule:
        input: [f"run/Taa/simulations/results_id{sid:03d}_{model}_{noise}.pkl" for (sid, model, noise) in chunk]
        group: "simgroup"
        output: touch(f"run/Taa/simulations/batches/{i}.done")



rule recordings:
    input:
        surf="data/Geometry/{subject}/surface.npz",
        contact_file="data/Geometry/{subject}/seeg.txt",
        rec_direc="data/Recordings/{subject}"
    output:
        results="run/Taa/recordings/results_{subject}.pkl",
        grouped="run/Taa/recordings/grouped_{subject}.pkl"
    resources:
        mem_mb=4096
    run:
        util.get_taa_in_recordings(wildcards.subject, input.surf, input.contact_file, input.rec_direc,
                                   output.results, output.grouped, plot='seizing')


rule aggreg_sim:
    input:
        batches=[f"run/Taa/simulations/batches/{i}.done" for i in range((len(SIDS)*len(MODELS)*len(NOISES)) // SIM_BATCH + 1)],
        grouped=expand("run/Taa/simulations/grouped_id{sid:03d}_{model}_{noise}.pkl", model=MODELS, noise=NOISES, sid=SIDS)
    output:
        grouped="run/Taa/df-grouped-sim.pkl"
    run:
        pd.concat([pd.read_pickle(fn) for fn in input.grouped], ignore_index=True).to_pickle(output.grouped)


rule aggreg_rec:
    input:
        results=expand("run/Taa/recordings/results_id{sid:03d}.pkl", sid=SIDS),
        grouped=expand("run/Taa/recordings/grouped_id{sid:03d}.pkl", sid=SIDS)
    output:
        results="run/Taa/df-results-rec.pkl",
        grouped="run/Taa/df-grouped-rec.pkl"
    run:
        pd.concat([pd.read_pickle(fn) for fn in input.results], ignore_index=True).to_pickle(output.results),
        pd.concat([pd.read_pickle(fn) for fn in input.grouped], ignore_index=True).to_pickle(output.grouped)


rule plot_rec_taas:
    input:
        df="run/Taa/df-grouped-rec.pkl",
        geom_direc="data/Geometry",
        rec_direc="data/Recordings/",
    output:
        imgdir=directory("run/Taa/recordings/groups/img"),
        mdfile="run/Taa/recordings/groups/TAA-groups.md"
    run:
        util.plot_groups(input.df, input.geom_direc, input.rec_direc, output.mdfile, output.imgdir)

rule md2pdf:
    input: "run/Taa/recordings/groups/TAA-groups.md"
    output: "run/Taa/recordings/groups/TAA-groups.pdf"
    shell: "pandoc {input} -o {output}"


rule all:
    input:
        simgrp="run/Taa/df-grouped-sim.pkl",
        recgrp="run/Taa/df-grouped-rec.pkl"
        # pdf="run/Taa/recordings/groups/TAA-groups.pdf"
