import requests
import xml.etree.ElementTree as ET
import json

def fetch_clinvar_variant_ids():
    # Define the base URL for ESearch
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    # Define the query parameters for ESearch
    esearch_params = {
        "db": "clinvar",
        "term": "clinsig pathogenic",
        "retmode": "json",
        "retmax": 10000  # Adjust as needed for more results
    }

    # Make the request to ESearch
    esearch_response = requests.get(esearch_url, params=esearch_params)
    if esearch_response.status_code != 200:
        raise Exception(f"ESearch request failed with status code {esearch_response.status_code}")

    # Extract the list of variant IDs from ESearch response
    esearch_data = esearch_response.json()
    variant_ids = esearch_data['esearchresult']['idlist']

    if not variant_ids:
        raise Exception("No variants found for the given query.")

    return variant_ids

def fetch_clinvar_variants(variant_ids_chunk):
    # Define the base URL for EFetch
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Define the query parameters for EFetch
    efetch_params = {
        "db": "clinvar",
        "id": ",".join(variant_ids_chunk),
        "retmode": "xml"
    }

    # Make the request to EFetch using POST
    efetch_response = requests.post(efetch_url, data=efetch_params)
    if efetch_response.status_code != 200:
        raise Exception(f"EFetch request failed with status code {efetch_response.status_code}")

    return efetch_response.text

def extract_variant_info(xml_data):
    root = ET.fromstring(xml_data)
    variants = []

    for clinvar_set in root.findall(".//ClinVarSet"):
        for measure_set in clinvar_set.findall(".//MeasureSet"):
            for measure in measure_set.findall(".//Measure"):
                seq_loc = measure.find(".//SequenceLocation")
                if seq_loc is not None:
                    chrom = seq_loc.get("Chr", None)
                    start = seq_loc.get("start", None)
                    ref_allele = seq_loc.get("referenceAllele", None)
                    alt_allele = seq_loc.get("alternateAllele", None)

                    if chrom and start and ref_allele and alt_allele:
                        variant_info = {
                            "chromosome": chrom,
                            "position": start,
                            "reference_allele": ref_allele,
                            "alternative_allele": alt_allele
                        }
                        variants.append(variant_info)

    return variants

def write_to_vcf(variants, filename="pathogenic_splice_variants.vcf"):
    with open(filename, 'w') as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Write variant data
        for variant in variants:
            f.write(f"{variant['chromosome']}\t{variant['position']}\t.\t{variant['reference_allele']}\t{variant['alternative_allele']}\t.\t.\t.\n")

def main():
    # Fetch the ClinVar variant IDs
    variant_ids = fetch_clinvar_variant_ids()

    all_variants = []
    chunk_size = 100  # Adjust the chunk size as needed to avoid URL length issues

    for i in range(0, len(variant_ids), chunk_size):
        variant_ids_chunk = variant_ids[i:i + chunk_size]
        xml_data = fetch_clinvar_variants(variant_ids_chunk)
        variants = extract_variant_info(xml_data)
        all_variants.extend(variants)

    # Save the data to a JSON file
    with open("pathogenic_splice_variants.json", "w") as f:
        json.dump(all_variants, f, indent=4)

    # Write the data to a VCF file
    write_to_vcf(all_variants)

    print("Data downloaded and saved successfully.")

if __name__ == "__main__":
    main()

