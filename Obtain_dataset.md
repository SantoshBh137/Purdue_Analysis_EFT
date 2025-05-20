#  Guide: How to Download CMS Datasets from DAS Using the Terminal

This guide explains how to locate, query, and download CMS NanoAOD datasets from the CMS Data Aggregation System (DAS) using a terminal environment. It also includes how to rename and organize downloaded ROOT files.

---

##  Step 1: Use the DAS Query Web Interface

To find your dataset, visit the DAS interface:
👉 [https://cmsweb.cern.ch/das/](https://cmsweb.cern.ch/das/)

Paste the dataset query in the search bar. For example:

```
file dataset=/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-20UL16APVJMENano_106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM
```

This will return a list of ROOT files associated with that dataset.

---

## ⚙ Step 2: Prepare Your CMSSW Environment

To interact with CMS data and tools, you need a valid CMSSW environment:

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel CMSSW_11_3_4  # or any valid release
cd CMSSW_11_3_4/src
cmsenv
```

Why? This sets up your environment with CMS-specific paths and software.

---

##  Step 3: Initialize a VOMS Proxy

To authenticate with CMS grid services (like DAS and xrootd), initialize your VOMS proxy:

```bash
voms-proxy-init --voms cms --valid 192:00
```

Why? This creates a short-lived certificate allowing you to download files from CMS data servers.

To check your proxy status:

```bash
voms-proxy-info
```

---

## 📁 Step 4: Create a Directory for Your Files

```bash
mkdir -p ~/NanoAOD_downloads/TTTo2L2Nu_UL16APV
cd ~/NanoAOD_downloads/TTTo2L2Nu_UL16APV
```

---

## 📄 Step 5: Query DAS for File List

Use `dasgoclient` to extract the first N file paths:

```bash
dasgoclient -query="file dataset=/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-20UL16APVJMENano_106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM" | head -n 15 > files.txt
```

This saves the first 15 file paths to a local text file `files.txt`.

---

##  Step 6: Create a Shell Script to Download and Rename Files

Create a file `download_powheg_ttbar.sh` with the following contents:

```bash
#!/bin/bash

prefix="root://xrootd-cms.infn.it/"
tag="TTTo2L2Nu_powheg_v1_2016preVFP_NanoAOD_"
count=1

while read -r file; do
    filename=$(basename "$file")
    echo "Downloading $file"
    xrdcp "${prefix}${file}" "$filename"

    newname="${tag}${count}.root"
    echo "Renaming $filename -> $newname"
    mv "$filename" "$newname"

    ((count++))
done < files.txt
```

* `files.txt` contains the list of file paths
* Each file is downloaded using `xrdcp`
* Each file is renamed using the specified `tag`

---

## ▶ Step 7: Make the Script Executable and Run It

```bash
chmod +x download_powheg_ttbar.sh
./download_powheg_ttbar.sh
```

Your files will be downloaded and renamed in the current directory.

---

##  Done!

You now have a reproducible and scalable way to retrieve datasets from DAS for your analysis.
