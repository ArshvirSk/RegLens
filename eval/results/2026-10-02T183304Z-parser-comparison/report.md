# Parser comparison report

- generated: 2026-10-02T18:33:04+00:00
- git commit: ec64f125eb5deeb9901e58b8c1a00f677c0ad62b
- experiment config_hash: e98261b916838c6f32d6c0e0db425a70560bf431848aa037efd2c2edac1d0eef
- corpus_version: 0.1.0
- documents: 20
- mean agreement (token Jaccard): 0.9732
- faster count: {'pymupdf': 20, 'pdfplumber': 0}

| doc_id | pages | pymupdf s | pdfplumber s | agreement | quality pymupdf/plumber |
|---|---:|---:|---:|---:|---:|
| rbi_md_kyc_2016 | 107 | 0.824 | 20.065 | 0.9503 | 1.0/1.0 |
| rbi_md_kyc_nbfc_2025 | 96 | 0.475 | 14.806 | 0.9995 | 1.0/1.0 |
| rbi_md_irac_2025 | 54 | 0.276 | 7.693 | 0.9996 | 1.0/1.0 |
| rbi_md_alm_2025 | 174 | 1.35 | 39.939 | 0.9927 | 1.0/1.0 |
| rbi_md_investment_2023 | 104 | 0.636 | 18.612 | 0.9397 | 1.0/1.0 |
| rbi_md_it_outsourcing_2023 | 31 | 0.207 | 5.274 | 0.9599 | 1.0/1.0 |
| rbi_md_it_governance_2023 | 26 | 0.189 | 5.186 | 0.9619 | 1.0/1.0 |
| rbi_md_digital_lending_2025 | 24 | 0.155 | 3.784 | 0.9612 | 1.0/1.0 |
| rbi_md_psl_2020 | 43 | 0.381 | 9.902 | 0.9048 | 1.0/1.0 |
| rbi_md_nbfc_p2p_2017 | 38 | 0.329 | 9.454 | 0.9596 | 1.0/1.0 |
| rbi_md_ppi_2021 | 40 | 0.325 | 9.183 | 0.9936 | 1.0/1.0 |
| rbi_md_securitisation_2021 | 61 | 0.497 | 10.689 | 0.9706 | 1.0/1.0 |
| rbi_md_credit_debit_card_2022 | 35 | 0.289 | 7.154 | 0.9635 | 1.0/1.0 |
| rbi_md_account_aggregator_2016 | 57 | 0.476 | 10.79 | 0.9592 | 0.9825/0.9825 |
| rbi_md_risk_interbank_2016 | 67 | 0.369 | 9.437 | 0.9924 | 1.0/1.0 |
| rbi_md_wilful_defaulters_2024 | 36 | 0.33 | 8.19 | 0.957 | 1.0/1.0 |
| rbi_md_credit_information_2025 | 92 | 0.651 | 18.401 | 0.9986 | 0.9783/0.9783 |
| rbi_md_kyc_commercial_2025 | 101 | 0.563 | 16.644 | 0.9993 | 1.0/1.0 |
| rbi_notif_integrated_ombudsman_2021 | 1 | 0.015 | 0.169 | 1.0 | 1.0/1.0 |
| rbi_scheme_integrated_ombudsman_2021 | 19 | 0.125 | 3.773 | 1.0 | 1.0/1.0 |

## Totals

- **pymupdf**: {'total_seconds': 8.462, 'total_chars': 2324379, 'total_pages_with_text': 1203, 'mean_quality': 0.998}
- **pdfplumber**: {'total_seconds': 229.145, 'total_chars': 2213152, 'total_pages_with_text': 1203, 'mean_quality': 0.998}

Timings are single wall-clock runs: indicative, not microbenchmarks. Pages listed in `scan_pages` yielded almost no text and are the OCR candidates.
