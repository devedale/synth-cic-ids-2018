# Advanced IP Translation Strategy (NAT-like Mapping) - v2

## The Problem & Constraints
The IP substitution logic must adhere strictly to network topology rules to ensure graph-based algorithms (like ip2vec) don't learn false topological relationships:
1. *Strict 1:1 Mapping:* Mathematical guarantee of no "birthday collisions". Each unique original IP gets a uniquely dedicated synthetic IP.
2. *Private/Public Segregation:* Private IPs remain Private; Public IPs remain Public.
3. *Malicious/Benign Segregation:* Attackers and Victims must not accidentally be placed in the same /24 subnet (LAN collision) during substitution.
4. *RFC 1918 Rules:* Handling class B private correctly (172.16.x.x to 172.31.x.x).

## Proposed Strategy: Dynamic Subnet-Aware Translation Table

We will build an exact dictionary in PySpark Driver memory per day before processing rows. 

### Step 1: Pre-computation (Finding Unique IPs & Roles)
python
# Identify all unique IPs and which ones acted as Attackers
all_ips_df = df.select("Src IP").union(df.select("Dst IP")).distinct()
unique_ips = [row[0] for row in all_ips_df.collect() if row[0]]

attack_cond = (F.trim(F.lower(F.col("Label"))) != "benign")
malicious_srcs = {row[0] for row in df.filter(attack_cond).select("Src IP").distinct().collect() if row[0]}


### Step 2: The Translation Table Engine
We will classify each unique IP into one of 4 categories and map it using mathematically disjoint pools to physically prevent LAN collisions between roles.

1. *Category 1: Public Malicious* 
   - *Source:* Pulled strictly from Threat Intel Feeds using set.pop(). (Guarantees zero duplicates).
2. *Category 2: Public Benign*
   - *Source:* Pulled strictly from Benign Feeds using set.pop().
3. *Category 3: Private Malicious*
   - *Risk:* Cannot share LAN with Private Benign.
   - *Solution:* We dedicate a specific RFC 1918 block purely for synthetic internal attackers, e.g., the 192.168.x.x/16 block. We generate unique IPs incrementally or randomly from this block.
4. *Category 4: Private Benign*
   - *Risk:* Must stay segregated.
   - *Solution:* We dedicate the massive 10.x.x.x/8 block (and 172.16.x.x - 172.31.x.x) purely for synthetic internal benign nodes. 

By hard-segregating the synthetic subnets for Private IPs based on role, it becomes *impossible* for a Private Malicious IP and a Private Benign IP to experience a LAN collision.
For Public IPs, reading from external curated feeds (Threat Intel) and popping exactly one IP ensures no internal IP-space collisions occur.

### Step 3: Fast Broadcast Mapping (PySpark create_map)
We broadcast this collision-free ip_map dictionary to the PySpark executors. We drop the Row-by-Row UDF entirely.

python
from itertools import chain
mapping_expr = F.create_map([F.lit(x) for x in chain(*ip_map.items())])

# In PySpark, map lookup is O(1) and executes in C/Java backend without Python UDF overhead.
df_processed = df.withColumn("Src IP", mapping_expr[F.col("Src IP")]) \
                 .withColumn("Dst IP", mapping_expr[F.col("Dst IP")])


## User Review Required
Queste modifiche ti convincono?
1. Usando insiemi (set) e la funzione pop(), è impossibile assegnare due volte lo stesso IP pubblico (niente collisioni del compleanno).
2. Per gli IP Privati (LAN), isoliamo gli attaccanti assegnando loro un blocco (es. 192.168.x.x) e le vittime un altro blocco (es. 10.x.x.x & 172.16-31.x.x). Questo rende impossibile una collisione di LAN tra attaccante e vittima.
3. Lo facciamo usando una mappa caricata in memoria, quindi l'esecuzione su Spark è istantanea.

Attenzione ad allineare bene il settings i requirements, README e tutti i file correlati