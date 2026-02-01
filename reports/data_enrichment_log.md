# Data Enrichment Log

**Project**: Ethiopia Financial Inclusion Forecasting
**Date**: 2026-02-01
**Collector**: Data Science Team

---

## Summary

Total enrichments: 17

- Observations: 8
- Events: 4
- Impact Links: 5

---

## Observations

### OBS_0001: Mobile Money Account Penetration

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_MM_ACCOUNT |
| **Value** | 6.8 |
| **Date** | 2022-12-31 |
| **Source** | NBE Annual Report 2022 |
| **Source URL** | https://www.nbe.gov.et |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | Interpolated between 2021 (4.7%) and 2024 (9.45%) based on growth trajectory |

### OBS_0002: Digital Payment Usage Rate

| Field | Value |
|-------|-------|
| **Indicator Code** | USG_DIGITAL_PAYMENT |
| **Value** | 18.5 |
| **Date** | 2022-12-31 |
| **Source** | World Bank Findex 2021 (extrapolated) |
| **Source URL** | https://www.worldbank.org/en/publication/globalfindex |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | Extrapolated from 2021 Findex data (17.2%) with growth trend |

### OBS_0003: Account Ownership Rate

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Value** | 48.3 |
| **Date** | 2024-12-31 |
| **Source** | World Bank Findex 2024 |
| **Source URL** | https://www.worldbank.org/en/publication/globalfindex |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Disaggregated data showing urban-rural gap in financial inclusion |

### OBS_0004: Account Ownership Rate

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Value** | 28.7 |
| **Date** | 2024-12-31 |
| **Source** | World Bank Findex 2024 |
| **Source URL** | https://www.worldbank.org/en/publication/globalfindex |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Disaggregated data showing significant urban-rural gap (19.6pp difference) |

### OBS_0005: Account Ownership Rate

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Value** | 32.1 |
| **Date** | 2024-12-31 |
| **Source** | World Bank Findex 2024 |
| **Source URL** | https://www.worldbank.org/en/publication/globalfindex |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Gender gap: 5.2pp lower than male (37.3%) |

### OBS_0006: Account Ownership Rate

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Value** | 37.3 |
| **Date** | 2024-12-31 |
| **Source** | World Bank Findex 2024 |
| **Source URL** | https://www.worldbank.org/en/publication/globalfindex |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Gender gap: 5.2pp higher than female (32.1%) |

### OBS_0007: Mobile Money Transaction Volume

| Field | Value |
|-------|-------|
| **Indicator Code** | USG_MM_VOLUME |
| **Value** | 850.0 |
| **Date** | 2023-12-31 |
| **Source** | NBE Payment Systems Report 2023 |
| **Source URL** | https://www.nbe.gov.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Significant growth from 2022, driven by Telebirr and M-Pesa expansion |

### OBS_0008: 4G Network Coverage

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_4G_COVERAGE |
| **Value** | 65.0 |
| **Date** | 2023-12-31 |
| **Source** | Ethio Telecom Annual Report 2023 |
| **Source URL** | https://www.ethiotelecom.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Infrastructure indicator critical for digital financial services |

## Events

### EVT_0011: M-Pesa Full Commercial Launch

| Field | Value |
|-------|-------|
| **Event Name** | M-Pesa Full Commercial Launch |
| **Date** | 2023-08-15 |
| **Source** | Safaricom Ethiopia Press Release |
| **Source URL** | https://www.safaricom.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Major mobile money product launch by Safaricom Ethiopia, expected to increase competition and adoption |

### EVT_0012: National Financial Inclusion Strategy 2023 Update

| Field | Value |
|-------|-------|
| **Event Name** | National Financial Inclusion Strategy 2023 Update |
| **Date** | 2023-06-01 |
| **Source** | National Bank of Ethiopia |
| **Source URL** | https://www.nbe.gov.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Policy framework update setting ambitious targets for financial inclusion |

### EVT_0013: Digital ID System (Fayda) Launch

| Field | Value |
|-------|-------|
| **Event Name** | Digital ID System (Fayda) Launch |
| **Date** | 2023-09-01 |
| **Source** | Ethiopian Digital ID Program |
| **Source URL** | https://www.fayda.gov.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Infrastructure milestone enabling KYC for financial services, reducing barriers to account opening |

### EVT_0014: Mobile Money Interoperability Framework

| Field | Value |
|-------|-------|
| **Event Name** | Mobile Money Interoperability Framework |
| **Date** | 2022-11-01 |
| **Source** | National Bank of Ethiopia |
| **Source URL** | https://www.nbe.gov.et |
| **Confidence** | high |
| **Collector** | Data Science Team |
| **Rationale** | Enables transfers between different mobile money providers, increasing utility and adoption |

## Impact Links

### IMP_0015: M-Pesa effect on Mobile Money Account Penetration

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_MM_ACCOUNT |
| **Impact Estimate** | 3.5pp |
| **Date** | None |
| **Evidence Basis** | empirical |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | Based on Kenya M-Pesa experience showing rapid adoption. Expected to add 3-4pp to mobile money penetration within 6-12 months |

### IMP_0016: M-Pesa effect on Digital Payment Usage

| Field | Value |
|-------|-------|
| **Indicator Code** | USG_DIGITAL_PAYMENT |
| **Impact Estimate** | 2.8pp |
| **Date** | None |
| **Evidence Basis** | literature |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | M-Pesa expected to drive digital payment adoption, especially in urban areas. Literature from Kenya shows 2-3pp increase in digital payment usage |

### IMP_0017: Digital ID effect on Account Ownership

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Impact Estimate** | 2.0pp |
| **Date** | None |
| **Evidence Basis** | expert |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | Digital ID reduces KYC barriers, enabling easier account opening. India Aadhaar experience shows 1-2pp increase in account ownership |

### IMP_0018: Interoperability effect on Digital Payment Usage

| Field | Value |
|-------|-------|
| **Indicator Code** | USG_DIGITAL_PAYMENT |
| **Impact Estimate** | 1.5pp |
| **Date** | None |
| **Evidence Basis** | literature |
| **Confidence** | medium |
| **Collector** | Data Science Team |
| **Rationale** | Interoperability increases utility of mobile money, driving usage. Ghana experience shows 1-2pp increase in transaction frequency |

### IMP_0019: Financial Inclusion Strategy effect on Account Ownership

| Field | Value |
|-------|-------|
| **Indicator Code** | ACC_OWNERSHIP |
| **Impact Estimate** | 0.8pp |
| **Date** | None |
| **Evidence Basis** | expert |
| **Confidence** | low |
| **Collector** | Data Science Team |
| **Rationale** | Policy frameworks have indirect effects through coordinated interventions. Estimated small positive impact over longer term |

---

## Notes

- All enrichments align with the unified schema expectations
- Confidence levels: high (official sources), medium (estimated/interpolated), low (expert judgment)
- Impact estimates are in percentage points (pp)
- Dates follow YYYY-MM-DD format
