
# Specto 0.2 - Data Collector

The precussor to the Specto model, this repo is designed to collect the nessecary data. 


## Code and Explanation

1. 'Pre-Pre Process Data.py' -> Designed to accept "species.txt" as an input file and filter molecules with {'O', 'C', 'H', 'N', 'Br', 'Cl', 'S', 'Si'} as acceptable elements. 

- Input -> 'species.txt' 
- Output -> 'filtered_molecules.csv'

2. 'Scrap SMILES, SELFIES, TAUT.py' -> Process the molecules, by determining their SMILES, SELFIES, Tautomer, and functional groups. The output is a ".yaml" file and the example format is "output file 1". 

- Input -> 'filtered_molecules.csv'
- Output -> 'all_data_SMARTS_FINAL_2024.yaml' (Current)

3. 'Scrap IR and MS.py' -> Using the NIST API, scrape the '.JDX' files for Mass and IR spectum. The paths for the IR and Mass jdx files are saved in a secondary yaml file as "output file 2". 

- Input -> 'all_data_SMARTS_FINAL_2024.yaml' (Current)
- Outputs -> '.JDX' files for mass and IR + 'all_data_MS&IR_PATHS_2.yaml' (Current)

4. 'NMR Data Scrap.py' -> Using Selenium Webdriver, the 1H-NMR and 13C-NMR is scrapped through nmrdb. The splitting patterns are saved in a '.yaml' file as "output file 3" and the screenshots are saved.

- Input -> 'all_data_SMARTS_FINAL_2024.yaml' (Current)
- Output -> 'nmr_results.yaml' + screenshots for 1H-NMR & 13C-NMR

5. 'Helper_Functions.py' -> Helper functions to read the jdx files and extract the data. 

- Input -> .JDX file
- Output -> JDX file data

6. 'JDX_Mass_Spec.py' -> Given the JDX data from files, two mass spectra plots are created. One without axes and one with axes. 

- Input -> JDX file data
- Output -> Mass spectra plot with axes + Mass spectra plot with no axes

7. 'Plots and Bins - Optim.py' -> Given JDX files, plots the IR and Mass spectrum plots (similar to 6). The plots are saved as 'png' with and without the axes. The IR data is binned and saved in a 'CSV' and the MS data is also saved as a 'CSV'. 
- Input -> 'all_data_SMARTS_FINAL_2024.yaml' (Current)
- Output -> Two images for MS and IR (with and without axes) + yaml file for the paths + "CSV" for data. 







## Output File Examples

1. Output File 1 -> YAML File

- 'all_data_SMARTS_FINAL_2024.yaml' (Current)
----

```yaml
100-00-5:
  Empirical Formula: C6H4ClNO2
  Functional Groups:
  - aromatics
  - alkyl halides
  - nitro
  SELFIES: '[O-1][N+1][=Branch1][C][=O][C][=C][C][=C][Branch1][C][Cl][C][=C][Ring1][#Branch1]'
  SMILES: '[O-][N+](=O)c1ccc(Cl)cc1'
  Tautomer: '[O]:[N](:[O]):[C]1:[C]:[C]:[C](-[Cl]):[C]:[C]:1_4_0'
```

---- 

2. Output File 2 -> YAML File
- 'all_data_MS&IR_PATHS_2.yaml'
----

```yaml
100-00-5:
  Path: /Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/ir/100-00-5.jdx
  Type: ir
100-07-2:
  Path: /Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/mass/100-07-2.jdx
  Type: mass
```
---- 

3. Output File 3 -> YAML File
- 'nmr_results.yaml'
----

```yaml
100-06-1:
  NMR Results:
    13C NMR: "13C NMR: \u03B4 27.7 (1C, s), 56.0 (1C, s), 114.3 (2C, s), 130.7 (2C,\
      \ s), 135.0 (1C, s), 159.8 (1C, s), 197.0 (1C, s)."
    1H NMR: "1H NMR: \u03B4 2.23 (3H, s), 3.79 (3H, s), 7.04 (2H, ddd, J = 8.3, 1.3,\
      \ 0.4 Hz), 7.97 (2H, ddd, J = 8.3, 1.6, 0.4 Hz)."
  Screenshots:
    13C NMR: /Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Screenshots/13C_NMR/100-06-1_13c_nmr.png
    1H NMR: /Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Screenshots/1H_NMR/100-06-1_1h_nmr.png
```
---- 

