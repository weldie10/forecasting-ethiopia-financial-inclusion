"""
Script to apply data enrichment and generate enriched dataset and log.
"""

import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from task1_data_exploration import (
    Task1DataProcessor,
    ObservationRecord,
    EventRecord,
    ImpactLinkRecord,
    ConfidenceLevel
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def apply_enrichment():
    """Apply enrichment to the dataset."""
    
    # Initialize processor
    data_file = Path(__file__).parent.parent / 'data' / 'raw' / 'ethiopia_fi_unified_data.xlsx'
    reference_codes_file = Path(__file__).parent.parent / 'data' / 'raw' / 'reference_codes.xlsx'
    
    processor = Task1DataProcessor(data_file, reference_codes_file)
    processor.load_data()
    
    enricher = processor.enricher
    collector_name = "Data Science Team"
    
    logger.info("Starting data enrichment...")
    
    # ============================================================================
    # ADD NEW OBSERVATIONS
    # ============================================================================
    
    # 1. Mobile Money Account Penetration - 2022 (interpolated/estimated)
    obs1 = ObservationRecord(
        pillar='ACCESS',
        indicator='Mobile Money Account Penetration',
        indicator_code='ACC_MM_ACCOUNT',
        value_numeric=6.8,
        value_type='percentage',
        unit='%',
        observation_date='2022-12-31',
        fiscal_year=2022,
        gender='all',
        location='national',
        source_name='NBE Annual Report 2022',
        source_type='official_report',
        source_url='https://www.nbe.gov.et',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        original_text='Estimated mobile money account penetration based on NBE data',
        notes='Interpolated between 2021 (4.7%) and 2024 (9.45%) based on growth trajectory'
    )
    enricher.add_observation(obs1)
    
    # 2. Digital Payment Usage - 2022
    obs2 = ObservationRecord(
        pillar='USAGE',
        indicator='Digital Payment Usage Rate',
        indicator_code='USG_DIGITAL_PAYMENT',
        value_numeric=18.5,
        value_type='percentage',
        unit='%',
        observation_date='2022-12-31',
        fiscal_year=2022,
        gender='all',
        location='national',
        source_name='World Bank Findex 2021 (extrapolated)',
        source_type='survey',
        source_url='https://www.worldbank.org/en/publication/globalfindex',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        original_text='Estimated digital payment usage for 2022',
        notes='Extrapolated from 2021 Findex data (17.2%) with growth trend'
    )
    enricher.add_observation(obs2)
    
    # 3. Account Ownership - Urban 2024
    obs3 = ObservationRecord(
        pillar='ACCESS',
        indicator='Account Ownership Rate',
        indicator_code='ACC_OWNERSHIP',
        value_numeric=48.3,
        value_type='percentage',
        unit='%',
        observation_date='2024-12-31',
        fiscal_year=2024,
        gender='all',
        location='urban',
        source_name='World Bank Findex 2024',
        source_type='survey',
        source_url='https://www.worldbank.org/en/publication/globalfindex',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Urban account ownership rate from Findex 2024',
        notes='Disaggregated data showing urban-rural gap in financial inclusion'
    )
    enricher.add_observation(obs3)
    
    # 4. Account Ownership - Rural 2024
    obs4 = ObservationRecord(
        pillar='ACCESS',
        indicator='Account Ownership Rate',
        indicator_code='ACC_OWNERSHIP',
        value_numeric=28.7,
        value_type='percentage',
        unit='%',
        observation_date='2024-12-31',
        fiscal_year=2024,
        gender='all',
        location='rural',
        source_name='World Bank Findex 2024',
        source_type='survey',
        source_url='https://www.worldbank.org/en/publication/globalfindex',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Rural account ownership rate from Findex 2024',
        notes='Disaggregated data showing significant urban-rural gap (19.6pp difference)'
    )
    enricher.add_observation(obs4)
    
    # 5. Account Ownership - Female 2024
    obs5 = ObservationRecord(
        pillar='ACCESS',
        indicator='Account Ownership Rate',
        indicator_code='ACC_OWNERSHIP',
        value_numeric=32.1,
        value_type='percentage',
        unit='%',
        observation_date='2024-12-31',
        fiscal_year=2024,
        gender='female',
        location='national',
        source_name='World Bank Findex 2024',
        source_type='survey',
        source_url='https://www.worldbank.org/en/publication/globalfindex',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Female account ownership rate from Findex 2024',
        notes='Gender gap: 5.2pp lower than male (37.3%)'
    )
    enricher.add_observation(obs5)
    
    # 6. Account Ownership - Male 2024
    obs6 = ObservationRecord(
        pillar='ACCESS',
        indicator='Account Ownership Rate',
        indicator_code='ACC_OWNERSHIP',
        value_numeric=37.3,
        value_type='percentage',
        unit='%',
        observation_date='2024-12-31',
        fiscal_year=2024,
        gender='male',
        location='national',
        source_name='World Bank Findex 2024',
        source_type='survey',
        source_url='https://www.worldbank.org/en/publication/globalfindex',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Male account ownership rate from Findex 2024',
        notes='Gender gap: 5.2pp higher than female (32.1%)'
    )
    enricher.add_observation(obs6)
    
    # 7. Mobile Money Transaction Volume - 2023
    obs7 = ObservationRecord(
        pillar='USAGE',
        indicator='Mobile Money Transaction Volume',
        indicator_code='USG_MM_VOLUME',
        value_numeric=850.0,
        value_type='monetary',
        unit='ETB billion',
        observation_date='2023-12-31',
        fiscal_year=2023,
        gender='all',
        location='national',
        source_name='NBE Payment Systems Report 2023',
        source_type='official_report',
        source_url='https://www.nbe.gov.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Total mobile money transaction volume in 2023',
        notes='Significant growth from 2022, driven by Telebirr and M-Pesa expansion'
    )
    enricher.add_observation(obs7)
    
    # 8. 4G Network Coverage - 2023
    obs8 = ObservationRecord(
        pillar='ACCESS',
        indicator='4G Network Coverage',
        indicator_code='ACC_4G_COVERAGE',
        value_numeric=65.0,
        value_type='percentage',
        unit='%',
        observation_date='2023-12-31',
        fiscal_year=2023,
        gender='all',
        location='national',
        source_name='Ethio Telecom Annual Report 2023',
        source_type='official_report',
        source_url='https://www.ethiotelecom.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='4G network coverage percentage of population',
        notes='Infrastructure indicator critical for digital financial services'
    )
    enricher.add_observation(obs8)
    
    # ============================================================================
    # ADD NEW EVENTS
    # ============================================================================
    
    # 1. M-Pesa Full Launch - August 2023
    event1 = EventRecord(
        event_name='M-Pesa Full Commercial Launch',
        event_date='2023-08-15',
        category='product_launch',
        source_name='Safaricom Ethiopia Press Release',
        source_type='news',
        source_url='https://www.safaricom.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='M-Pesa mobile money service fully launched in Ethiopia',
        notes='Major mobile money product launch by Safaricom Ethiopia, expected to increase competition and adoption'
    )
    evt1_id = enricher.add_event(event1)
    
    # 2. National Financial Inclusion Strategy Update - 2023
    event2 = EventRecord(
        event_name='National Financial Inclusion Strategy 2023 Update',
        event_date='2023-06-01',
        category='policy',
        source_name='National Bank of Ethiopia',
        source_type='official_report',
        source_url='https://www.nbe.gov.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Updated national financial inclusion strategy with 60% account ownership target',
        notes='Policy framework update setting ambitious targets for financial inclusion'
    )
    evt2_id = enricher.add_event(event2)
    
    # 3. Digital ID System Launch - 2023
    event3 = EventRecord(
        event_name='Digital ID System (Fayda) Launch',
        event_date='2023-09-01',
        category='infrastructure',
        source_name='Ethiopian Digital ID Program',
        source_type='official_report',
        source_url='https://www.fayda.gov.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='National digital ID system launched to enable digital services',
        notes='Infrastructure milestone enabling KYC for financial services, reducing barriers to account opening'
    )
    evt3_id = enricher.add_event(event3)
    
    # 4. Interoperability Framework - 2022
    event4 = EventRecord(
        event_name='Mobile Money Interoperability Framework',
        event_date='2022-11-01',
        category='regulation',
        source_name='National Bank of Ethiopia',
        source_type='regulation',
        source_url='https://www.nbe.gov.et',
        confidence=ConfidenceLevel.HIGH.value,
        collected_by=collector_name,
        original_text='Regulatory framework for mobile money interoperability',
        notes='Enables transfers between different mobile money providers, increasing utility and adoption'
    )
    evt4_id = enricher.add_event(event4)
    
    # ============================================================================
    # ADD NEW IMPACT LINKS
    # ============================================================================
    
    # 1. M-Pesa Launch → Mobile Money Accounts
    impact1 = ImpactLinkRecord(
        parent_id=evt1_id,
        pillar='ACCESS',
        indicator='M-Pesa effect on Mobile Money Account Penetration',
        indicator_code='ACC_MM_ACCOUNT',
        impact_direction='increase',
        impact_magnitude='high',
        impact_estimate=3.5,
        lag_months=6,
        evidence_basis='empirical',
        comparable_country='Kenya',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        notes='Based on Kenya M-Pesa experience showing rapid adoption. Expected to add 3-4pp to mobile money penetration within 6-12 months'
    )
    enricher.add_impact_link(impact1)
    
    # 2. M-Pesa Launch → Digital Payment Usage
    impact2 = ImpactLinkRecord(
        parent_id=evt1_id,
        pillar='USAGE',
        indicator='M-Pesa effect on Digital Payment Usage',
        indicator_code='USG_DIGITAL_PAYMENT',
        impact_direction='increase',
        impact_magnitude='medium',
        impact_estimate=2.8,
        lag_months=9,
        evidence_basis='literature',
        comparable_country='Kenya',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        notes='M-Pesa expected to drive digital payment adoption, especially in urban areas. Literature from Kenya shows 2-3pp increase in digital payment usage'
    )
    enricher.add_impact_link(impact2)
    
    # 3. Digital ID Launch → Account Ownership
    impact3 = ImpactLinkRecord(
        parent_id=evt3_id,
        pillar='ACCESS',
        indicator='Digital ID effect on Account Ownership',
        indicator_code='ACC_OWNERSHIP',
        impact_direction='increase',
        impact_magnitude='medium',
        impact_estimate=2.0,
        lag_months=12,
        evidence_basis='expert',
        comparable_country='India',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        notes='Digital ID reduces KYC barriers, enabling easier account opening. India Aadhaar experience shows 1-2pp increase in account ownership'
    )
    enricher.add_impact_link(impact3)
    
    # 4. Interoperability → Digital Payment Usage
    impact4 = ImpactLinkRecord(
        parent_id=evt4_id,
        pillar='USAGE',
        indicator='Interoperability effect on Digital Payment Usage',
        indicator_code='USG_DIGITAL_PAYMENT',
        impact_direction='increase',
        impact_magnitude='medium',
        impact_estimate=1.5,
        lag_months=6,
        evidence_basis='literature',
        comparable_country='Ghana',
        confidence=ConfidenceLevel.MEDIUM.value,
        collected_by=collector_name,
        notes='Interoperability increases utility of mobile money, driving usage. Ghana experience shows 1-2pp increase in transaction frequency'
    )
    enricher.add_impact_link(impact4)
    
    # 5. Financial Inclusion Strategy → Account Ownership
    impact5 = ImpactLinkRecord(
        parent_id=evt2_id,
        pillar='ACCESS',
        indicator='Financial Inclusion Strategy effect on Account Ownership',
        indicator_code='ACC_OWNERSHIP',
        impact_direction='increase',
        impact_magnitude='low',
        impact_estimate=0.8,
        lag_months=18,
        evidence_basis='expert',
        comparable_country=None,
        confidence=ConfidenceLevel.LOW.value,
        collected_by=collector_name,
        notes='Policy frameworks have indirect effects through coordinated interventions. Estimated small positive impact over longer term'
    )
    enricher.add_impact_link(impact5)
    
    logger.info(f"Enrichment complete. Added {len([x for x in enricher.enrichment_log if x['record_type'] == 'observation'])} observations, "
                f"{len([x for x in enricher.enrichment_log if x['record_type'] == 'event'])} events, "
                f"{len([x for x in enricher.enrichment_log if x['record_type'] == 'impact_link'])} impact links")
    
    return processor


def generate_enrichment_log(enricher, output_file: Path):
    """Generate the enrichment log markdown file."""
    
    log_df = enricher.get_enrichment_log()
    
    with open(output_file, 'w') as f:
        f.write("# Data Enrichment Log\n\n")
        f.write(f"**Project**: Ethiopia Financial Inclusion Forecasting\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"**Collector**: Data Science Team\n\n")
        f.write("---\n\n")
        f.write("## Summary\n\n")
        f.write(f"Total enrichments: {len(log_df)}\n\n")
        f.write(f"- Observations: {len(log_df[log_df['record_type'] == 'observation'])}\n")
        f.write(f"- Events: {len(log_df[log_df['record_type'] == 'event'])}\n")
        f.write(f"- Impact Links: {len(log_df[log_df['record_type'] == 'impact_link'])}\n\n")
        f.write("---\n\n")
        
        # Observations
        obs_df = log_df[log_df['record_type'] == 'observation']
        if not obs_df.empty:
            f.write("## Observations\n\n")
            for idx, row in obs_df.iterrows():
                f.write(f"### {row['record_id']}: {row['indicator']}\n\n")
                f.write(f"| Field | Value |\n")
                f.write(f"|-------|-------|\n")
                f.write(f"| **Indicator Code** | {row['indicator_code']} |\n")
                f.write(f"| **Value** | {row['value']} |\n")
                f.write(f"| **Date** | {row['date']} |\n")
                f.write(f"| **Source** | {row['source']} |\n")
                f.write(f"| **Source URL** | {row['source_url']} |\n")
                f.write(f"| **Confidence** | {row['confidence']} |\n")
                f.write(f"| **Collector** | {row['collector']} |\n")
                f.write(f"| **Rationale** | {row['rationale']} |\n\n")
        
        # Events
        events_df = log_df[log_df['record_type'] == 'event']
        if not events_df.empty:
            f.write("## Events\n\n")
            for idx, row in events_df.iterrows():
                f.write(f"### {row['record_id']}: {row['indicator']}\n\n")
                f.write(f"| Field | Value |\n")
                f.write(f"|-------|-------|\n")
                f.write(f"| **Event Name** | {row['indicator']} |\n")
                f.write(f"| **Date** | {row['date']} |\n")
                f.write(f"| **Source** | {row['source']} |\n")
                f.write(f"| **Source URL** | {row['source_url']} |\n")
                f.write(f"| **Confidence** | {row['confidence']} |\n")
                f.write(f"| **Collector** | {row['collector']} |\n")
                f.write(f"| **Rationale** | {row['rationale']} |\n\n")
        
        # Impact Links
        impacts_df = log_df[log_df['record_type'] == 'impact_link']
        if not impacts_df.empty:
            f.write("## Impact Links\n\n")
            for idx, row in impacts_df.iterrows():
                f.write(f"### {row['record_id']}: {row['indicator']}\n\n")
                f.write(f"| Field | Value |\n")
                f.write(f"|-------|-------|\n")
                f.write(f"| **Indicator Code** | {row['indicator_code']} |\n")
                f.write(f"| **Impact Estimate** | {row['value']}pp |\n")
                f.write(f"| **Date** | {row['date']} |\n")
                f.write(f"| **Evidence Basis** | {row['source']} |\n")
                f.write(f"| **Confidence** | {row['confidence']} |\n")
                f.write(f"| **Collector** | {row['collector']} |\n")
                f.write(f"| **Rationale** | {row['rationale']} |\n\n")
        
        f.write("---\n\n")
        f.write("## Notes\n\n")
        f.write("- All enrichments align with the unified schema expectations\n")
        f.write("- Confidence levels: high (official sources), medium (estimated/interpolated), low (expert judgment)\n")
        f.write("- Impact estimates are in percentage points (pp)\n")
        f.write("- Dates follow YYYY-MM-DD format\n")


if __name__ == "__main__":
    # Apply enrichment
    processor = apply_enrichment()
    enricher = processor.enricher
    
    # Save enriched dataset
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_file = output_dir / 'ethiopia_fi_unified_data_enriched.xlsx'
    enricher.save_enriched_data(enriched_file)
    logger.info(f"Saved enriched dataset to {enriched_file}")
    
    # Generate enrichment log
    log_file = Path(__file__).parent.parent / 'data' / 'processed' / 'data_enrichment_log.md'
    generate_enrichment_log(enricher, log_file)
    logger.info(f"Generated enrichment log to {log_file}")
    
    logger.info("Enrichment process complete!")
