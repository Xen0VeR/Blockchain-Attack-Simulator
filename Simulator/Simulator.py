import argparse
import random
import heapq
import matplotlib.pyplot as plt
from graphviz import Digraph
import seaborn as sns
import pandas as pd
from collections import deque
import itertools
import networkx as nx
import numpy as np
from collections import defaultdict

# ----------------------------
# Event Types
# ----------------------------

TXN_GENERATE = "TXN_GENERATE"
TXN_RECEIVE = "TXN_RECEIVE"
SEND_BLOCK = "SEND_BLOCK"
RECEIVE_BLOCK = "RECEIVE_BLOCK"

# Counters for unique IDs for block and transaction
blk_counter = itertools.count(1)
txn_counter = itertools.count(1) 

# ------------------------------
# Peer Class 
# ------------------------------

class Peer:
    def __init__(self, n, z0, z1):
        self.n = n
        self.z0 = z0
        self.z1 = z1

    class Node:
        def __init__(self, peer_id, speed, cpu):
            self.peer_id = peer_id
            self.speed = speed
            self.cpu = cpu
            self.hash_power = 1 if cpu == 'low CPU' else 10
            self.hash_fraction = 0 
            self.balance = 100  #### CHANGEABLE ####
            self.pending_txns = [] 
            self.block_tree = {}  # For blockchain structure
            self.balance_map = {} # dict : peer_id -> balance (update the map on recieving block)

        # representation of node
        def __repr__(self):
            return f"Peer({self.peer_id}, {self.speed}, {self.cpu})"

    def generate_peers(self):
        num_slow = int((self.z0 / 100) * self.n)
        num_low_cpu = int((self.z1 / 100) * self.n)

        slow_ids = set(random.sample(range(0, self.n), num_slow))
        low_cpu_ids = set(random.sample(range(0, self.n), num_low_cpu))

        peers = []

        for i in range(0, self.n):
            speed = 'slow' if i in slow_ids else 'fast'
            cpu = 'low CPU' if i in low_cpu_ids else 'high CPU'
            peers.append(self.Node(i, speed, cpu))
        return peers

# ------------------------------
# Transaction Class
# ------------------------------

class Transaction:
    def __init__(self, txn_id, sender_id, receiver_id, amount):
        self.txn_id = txn_id
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.amount = amount

    # validity of transaction
    def isValid(self,sender_balance):
        if self.amount <= 0:
            return False
        if self.sender_id == self.receiver_id:
            return False
        if sender_balance < self.amount:
            return False
        return True
    
    # representation of transaction
    def __repr__(self):
        return f"{self.txn_id}: {self.sender_id} pays {self.receiver_id} {self.amount} coins"
    
# ------------------------------
# Block Class
# ------------------------------

class Block:
    def __init__(self, blk_id, parent_id, transactions, creator_id):
        self.blk_id = blk_id
        self.parent_id = parent_id
        self.transactions = transactions
        self.creator_id = creator_id

# ------------------------------
# Event Class
# ------------------------------

class Event:
    def __init__(self, timestamp, event_type, peer_id):
        self.timestamp = timestamp
        self.event_type = event_type
        self.peer_id = peer_id
        self.txn = None # Transaction associated to the event (For TXN_RECEIVE : receiver needs to store the transaction in pending transactions) 

    def __lt__(self, other):
        return self.timestamp < other.timestamp

# ------------------------------
# Event Queue Class
# ------------------------------

class EventQueue:
    def __init__(self):
        self.queue = []

    def add_event(self, event):
        heapq.heappush(self.queue, event)

    def pop_event(self):
        return heapq.heappop(self.queue) if self.queue else None

    def is_empty(self):
        return len(self.queue) == 0 

# ------------------------------
# Network Class
# ------------------------------

class Network:
    def __init__(self, n, z0, z1):
        self.n = n
        self.peers = Peer(n, z0, z1).generate_peers()
        self.adjacency_list = {i: set() for i in range(0,self.n)}
        self.visited = set() # for BFS to check connectivity
        self.graph = nx.Graph()
        self.latencies = {}
        self.link_speeds = {} 

        # Normalising the hash power
        total_power = sum(p.hash_power for p in self.peers)
        for p in self.peers:
            p.hash_fraction = p.hash_power/total_power  # This is h_k 

    def create_adjacency_list(self):
        max_attempts = 10000 # max number of attempts to create adjacency list
        attempts = 0
        while not all(len(neighbours) >= 3 for neighbours in self.adjacency_list.values()):
            if attempts >= max_attempts:
                self.adjacency_list = {i: set() for i in range(self.n)}
                attempts = 0
            i = random.randint(0, self.n - 1)
            j = random.randint(0, self.n - 1)
            if i != j and len(self.adjacency_list[i]) < 6 and len(self.adjacency_list[j]) < 6:
                if j not in self.adjacency_list[i]:
                    self.adjacency_list[i].add(j)
                    self.adjacency_list[j].add(i)
            attempts += 1

    def BFS(self, source): # helper in checking connectivity
        q = deque([source])
        self.visited.add(source)
        while q:
            current = q.popleft()
            for neighbour in self.adjacency_list[current]:
                if neighbour not in self.visited:
                    self.visited.add(neighbour)
                    q.append(neighbour)

    def isConnected(self):
        connected = False
        while not connected:
            self.create_adjacency_list()
            self.visited.clear()
            count = 0
            for node in range(0,self.n):
                if node not in self.visited:
                    count += 1
                    if count > 1:
                        break
                    self.BFS(node)
            if count == 1:
                connected = True
            # reset the adjacency list 
            if not connected:
                self.adjacency_list = {i: set() for i in range(0,self.n)}
        return True

    def degree_check(self): # degree constraints : 3 <= degree(v) <= 6
        for neighbours in self.adjacency_list.values():
            if len(neighbours) < 3 or len(neighbours) > 6:
                return False
        return True   

    def generate_topology(self):
        for node,neighbours in self.adjacency_list.items():
            for neighbour in neighbours:
                self.graph.add_edge(node,neighbour)
        return self.graph

    def initialize_latencies_and_speeds(self): 
        for i in self.adjacency_list:
            for j in self.adjacency_list[i]:
                if (j,i) in self.latencies: 
                    continue
                rho = random.uniform(10,500)/1000 # speed of light propogation delay in seconds
                self.latencies[(i,j)] = rho
                self.latencies[(j,i)] = rho

                peer_i = self.peers[i] 
                peer_j = self.peers[j]
                if peer_i.speed == 'fast' and peer_j.speed == 'fast':
                    c = 100 * 1e6 # 100Mbps 
                else:
                    c = 5 * 1e6 # 5 Mbps

                self.link_speeds[(i,j)] = c  # link speed between i and j in bits/second
                self.link_speeds[(j,i)] = c


    def compute_latency(self, i, j, message_size_bits):
        # Latency = ρ + |m| / c + d 
        if (i, j) not in self.latencies or (i, j) not in self.link_speeds:
            # No direct link: return infinite latency
            print(f"Warning: No direct link between Peer {i} and Peer {j}")
            return float('inf')
        
        rho = self.latencies[(i,j)]
        c = self.link_speeds[(i,j)]
        second_term = message_size_bits/c  # |m|/c_{i,j}
        d_mean = (96*1024)/c 
        d = np.random.exponential(d_mean) # d_{i,j} is queuing delay at node i to forward message to node j

        return rho + second_term + d
    
# ------------------------------
# BlockchainTree Class
# ------------------------------

class BlockchainTree:
    def __init__(self):
        self.blocks = {}  # dict : blk_id -> block 
        self.children = {} # dict : blk_id -> children_blk_ids
        self.block_times = {} # dict : blk_id -> time of arrival of block
        self.longest_chain_tip = None # blk_id corresponding to tip of longest chain
        self.orphan_blocks = defaultdict(list)

    def add_block(self,block,time):
        self.blocks[block.blk_id] = block
        self.block_times[block.blk_id] = time
        if block.parent_id is not None: 
            self.children.setdefault(block.parent_id, []).append(block.blk_id) # stores the id's of the children block
        if self.longest_chain_tip is None or self.chain_length(block.blk_id) > self.chain_length(self.longest_chain_tip):
            self.longest_chain_tip = block.blk_id

    def chain_length(self,blk_id):
        length = 0
        while blk_id:
            blk_id = self.blocks[blk_id].parent_id
            length += 1
        return length

    def get_longest_chain(self):
        chain = []
        blk_id = self.longest_chain_tip
        while blk_id:
            chain.append(blk_id)
            blk_id = self.blocks[blk_id].parent_id
        return list(reversed(chain)) # From genesis to tip

    def get_branch_lengths(self,miner_id): # branches in the block_tree of miner_id block
        if self.longest_chain_tip is None: # No blocks in block_tree
            return []
        
        longest_chain = set(self.get_longest_chain())
        branch_lengths = []

        # leaf blocks (blocks with no children)
        leaf_blocks = [blk_id for blk_id in self.blocks if blk_id not in self.children]

        for leaf in leaf_blocks:
            if leaf in longest_chain:
                continue # skip longest chain tip
            block = self.blocks[leaf]
            if block.creator_id != miner_id:
                continue
            length = self.chain_length(leaf)
            branch_lengths.append(length)

        return branch_lengths
            
    def get_chain_txns(self,blk_id): # transactions in a chain with blk_id at the tip of the chain
        txns = []
        while blk_id:
            txns.extend(self.blocks[blk_id].transactions)
            blk_id = self.blocks[blk_id].parent_id
        return set(t.txn_id for t in txns)

# ------------------------------
# Simulator Class
# ------------------------------

class Simulator:
    def __init__(self, n, z0, z1, t_tx, block_interval):
        self.n = n
        self.z0 = z0
        self.z1 = z1
        self.t_tx = t_tx
        self.block_interval = block_interval
        self.event_queue = EventQueue()
        self.network = Network(n, z0, z1)
        self.current_time = 0

    def initialize(self):
        # generate genesis block, schedule initial events
        print("Initializing network...")
        self.network.isConnected() # Generate the adjacency list
        self.network.initialize_latencies_and_speeds()

        # Schedule first transaction generation for all peers
        for peer in self.network.peers:
            next_time = np.random.exponential(self.t_tx)
            event = Event(timestamp=next_time, event_type=TXN_GENERATE, peer_id=peer.peer_id)
            self.event_queue.add_event(event)

        # Each Peer maintains a balance map of all the peers
        for peer in self.network.peers:
            for p in self.network.peers:
                peer.balance_map[p.peer_id] = p.balance

        # Adding Genesis Block 
        genesis_block = Block("blk_0",None,[],-1) # Creator is -1 for genesis block
        for peer in self.network.peers:
            peer.block_tree = BlockchainTree()
            peer.block_tree.add_block(genesis_block,0)

        # schedule mining for each peer
        for peer in self.network.peers:
            self.schedule_block_mining(peer, self.current_time)

        print("Initialization Complete .")

    def run(self, end_time):
        while not self.event_queue.is_empty() and self.current_time <= end_time:
            event = self.event_queue.pop_event()
            self.current_time = event.timestamp
            self.handle_event(event)

    def handle_event(self, event):
        # TODO: handle txn creation, propagation, block minig, block receiving 
        if event.event_type == TXN_GENERATE:
            self.handle_txn_generation(event.peer_id)
        elif event.event_type == TXN_RECEIVE:
            self.handle_txn_receive(event.peer_id,event.txn)
        elif event.event_type == SEND_BLOCK:
            self.handle_block_creation(event.peer_id)
        elif event.event_type == RECEIVE_BLOCK:
            self.handle_block_receive(event.peer_id, event.block)
    
    def handle_txn_generation(self,peer_id):
        peer = self.network.peers[peer_id] 
        receiver_id = random.choice([p.peer_id for p in self.network.peers if p.peer_id != peer_id])
        amount = random.randint(1,10)  ### CHANGEABLE ### 

        txn_id = f"txn_{next(txn_counter)}"
        txn = Transaction(txn_id,peer_id,receiver_id,amount)

        if txn.isValid(peer.balance):
            print(f"[{self.current_time:.2f}s] Peer {peer_id} generated transaction: {txn}")
            peer.pending_txns.append(txn) # add the transaction in it's own pending transactions

            # Broadcast to the neighbours
            for neighbour_id in self.network.adjacency_list[peer_id]:
                latency = self.network.compute_latency(peer_id,neighbour_id,message_size_bits=8*1024)  # 1KB message
                receive_time = self.current_time + latency
                event = Event(timestamp=receive_time,event_type=TXN_RECEIVE,peer_id=neighbour_id)
                event.txn = txn  # Attach transaction with the event so the receiver can add the transaction in it's pending transactions
                self.event_queue.add_event(event)
        
        else:
            print(f"[{self.current_time:.2f}s] Peer {peer_id} could NOT generate transaction (invalid): {txn}")

        # Schedule next transaction generation
        next_time = self.current_time + np.random.exponential(self.t_tx)
        self.event_queue.add_event(Event(next_time, TXN_GENERATE, peer_id))

    def handle_txn_receive(self,peer_id,txn):
        peer = self.network.peers[peer_id] # receiver

        # Check if transaction already received (i.e. in pending_txns)
        if txn.txn_id in [t.txn_id for t in peer.pending_txns]:
            return
        
        # Store it
        peer.pending_txns.append(txn)

        print(f"[{self.current_time:.2f}s] Peer {peer_id} received transaction: {txn}")

        # Forward to neighbour (except the sender)
        for neighbour_id in self.network.adjacency_list[peer_id]:
            if neighbour_id == txn.sender_id:
                continue

            latency = self.network.compute_latency(peer_id,neighbour_id,message_size_bits=8*1024)
            receive_time = self.current_time + latency

            event  = Event(timestamp=receive_time,event_type=TXN_RECEIVE,peer_id=neighbour_id)
            event.txn = txn
            self.event_queue.add_event(event)

    def schedule_block_mining(self,peer,current_time):
        hk = peer.hash_fraction
        Tk = np.random.exponential(self.block_interval/hk) 
        event = Event(timestamp=current_time + Tk, event_type=SEND_BLOCK, peer_id=peer.peer_id)
        self.event_queue.add_event(event)

    def handle_block_creation(self,peer_id):
        peer = self.network.peers[peer_id]
        tip = peer.block_tree.longest_chain_tip
        seen_txns = peer.block_tree.get_chain_txns(tip) # transaction till now in the longest chain of the peer block tree --> should not be included in the new block

        block_txns = []
        total_size = 1024 # 1KB coinbase
        for txn in peer.pending_txns:
            if txn.txn_id not in seen_txns:
                txn_size = 1024
                if  total_size + txn_size > 8000000:  # 1MB = 8*10^6
                    break
                if txn.isValid(self.network.peers[txn.sender_id].balance):
                    block_txns.append(txn)
                    total_size += txn_size
        
        coinbase = Transaction(f"coinbase_{peer_id}_{self.current_time:.2f}",peer_id,peer_id,50)
        block_txns.insert(0,coinbase) # insert coinbase transaction at top of the new block
 
        blk_id = f"blk_{next(blk_counter)}"
        new_block = Block(blk_id,tip,block_txns,peer_id)
        peer.block_tree.add_block(new_block,self.current_time)

        peer.pending_txns = [txn for txn in peer.pending_txns if txn not in block_txns] # remove the transactions added in the new block from the pending transactions

        print(f"[{self.current_time:.2f}s] Peer {peer_id} mined block {blk_id} with {len(block_txns)} txns")

        # Update balance
        for txn in block_txns:
            if txn.sender_id != txn.receiver_id: # for coinbase transaction txn.sender_id == txn.receiver_id . hence it should only be added not deducted
                peer.balance_map[txn.sender_id] -= txn.amount
            peer.balance_map[txn.receiver_id] += txn.amount

        # Broadcast to neighbours
        for neighbour_id in self.network.adjacency_list[peer_id]:
            latency = self.network.compute_latency(peer_id,neighbour_id,message_size_bits=8*total_size)  # total_size is in bytes
            receive_time = self.current_time + latency
            event = Event(receive_time,RECEIVE_BLOCK,neighbour_id)
            event.block = new_block
            self.event_queue.add_event(event)


    def handle_block_receive(self,peer_id,block):
        peer = self.network.peers[peer_id]

        # Check if already seen
        if block.blk_id in peer.block_tree.blocks:
            return

        # If parent is missing, store as orphan and return
        if block.parent_id and block.parent_id not in peer.block_tree.blocks:
            print(f"[{self.current_time:.2f}s] Peer {peer_id} received orphan block {block.blk_id} (missing parent {block.parent_id})")
            peer.block_tree.orphan_blocks[block.parent_id].append(block)
            return
        
        # Validate transactions
        temp_balance = peer.balance_map.copy()
        valid = True
        for txn in block.transactions:
            if txn.sender_id != txn.receiver_id:
                if txn.amount <= 0 or temp_balance[txn.sender_id]<txn.amount:
                    valid = False
                    break
                temp_balance[txn.sender_id] -= txn.amount
                temp_balance[txn.receiver_id] += txn.amount
            else:
                temp_balance[txn.receiver_id] += txn.amount # coinbase transaction . Should only be added into the miner's account not deducted from anyone

        if not valid:
            print(f"[{self.current_time:.2f}s] Peer {peer_id} rejected block {block.blk_id} (invalid txns)")
            return
        
        # Update balances
        peer.block_tree.add_block(block,self.current_time)
        if peer.block_tree.longest_chain_tip == block.blk_id:
            # Apply temp_balance updates to actual peer.balance
            peer.balance_map = temp_balance.copy()
            self.schedule_block_mining(peer, self.current_time)

        print(f"[{self.current_time:.2f}s] Peer {peer_id} accepted block {block.blk_id}")

        # Check if any orphan blocks can now be added
        blk_id = block.blk_id
        if blk_id in peer.block_tree.orphan_blocks:
            orphans_to_process = peer.block_tree.orphan_blocks.pop(blk_id, [])
            for orphan in orphans_to_process:
                self.handle_block_receive(peer_id, orphan)

        # Restart mining if new longest chain
        if peer.block_tree.longest_chain_tip ==block.blk_id:
            self.schedule_block_mining(peer,self.current_time)
        
        # Propagate to neighbors
        for neighbour_id in self.network.adjacency_list[peer_id]:
            latency = self.network.compute_latency(peer_id, neighbour_id, message_size_bits=8*len(block.transactions)*1024)
            receive_time = self.current_time + latency
            event = Event(receive_time, RECEIVE_BLOCK, neighbour_id)
            event.block = block
            self.event_queue.add_event(event)

# ------------------------------
# Dump Block Trees to file
# ------------------------------

def dump_block_trees(sim):
    for peer in sim.network.peers:
        filename = f"peer_{peer.peer_id}_block_tree.txt"
        with open(filename,"w") as f:
            f.write("block_id\tparent_id\tcreater\tarrival_time\n")
            for blk_id in peer.block_tree.blocks:
                block = peer.block_tree.blocks[blk_id]
                arrival_time = peer.block_tree.block_times.get(blk_id,-1)
                f.write(f"{blk_id}\t{block.parent_id}\t{block.creator_id}\t{arrival_time:.2f}\n")

# ------------------------------
# Visualizations
# ------------------------------
class BlockchainAnalyzer:
    def __init__(self, sim):
        self.sim = sim
        self.ratios = {}
        self.branch_lengths = {}
    
    def analyze_longest_chain_contribution(self):
        peer_type = {}
        block_counts = defaultdict(int) 
        longest_counts = defaultdict(int)

        for peer in self.sim.network.peers:
            peer_type[peer.peer_id] = (peer.speed,peer.cpu)
            longest_chain = set(peer.block_tree.get_longest_chain())

            for blk_id,block in peer.block_tree.blocks.items():
                creator = block.creator_id
                if creator == -1:
                    continue # skip genesis block
                block_counts[creator] += 1
                if blk_id in longest_chain:
                    longest_counts[creator] += 1

        for peer_id in block_counts:
            total = block_counts[peer_id]
            longest = longest_counts.get(peer_id,0) # if peer_id not in longest_counts then returns 0
            speed, cpu = peer_type[peer_id]
            self.ratios[peer_id] = {
                'ratio' : longest / total if total > 0 else 0,
                'speed' : speed,
                'cpu' : cpu
            }

    def analyze_branch_lengths(self):
        for peer in self.sim.network.peers:
            lengths = peer.block_tree.get_branch_lengths(peer.peer_id)
            self.branch_lengths[peer.peer_id] = {
                'lengths' : lengths,
                'speed' : peer.speed,
                'cpu' : peer.cpu
            }
        

    def plot_longest_chain_ratios(self):
        peer_ids = []
        ratios = []
        speeds = []
        cpus = []

        for pid, data in sorted(self.ratios.items()):
            peer_ids.append(pid)
            ratios.append(data['ratio'])
            speeds.append(data['speed'])
            cpus.append(data['cpu'])

        plt.figure(figsize=(12, 6))

        # Group by (CPU, Speed) tuple
        group_keys = list(set(zip(cpus, speeds)))
        for cpu, speed in group_keys:
            x = [pid for pid, c, s in zip(peer_ids, cpus, speeds) if c == cpu and s == speed]
            y = [r for r, c, s in zip(ratios, cpus, speeds) if c == cpu and s == speed]
            label = f'{cpu}, {speed}'
            plt.plot(x, y, label=label)

        plt.title("Longest Chain Ratio per Peer (by CPU and Speed)")
        plt.xlabel("Peer ID")
        plt.ylabel("Ratio of Blocks in Longest Chain")
        plt.legend()
        plt.tight_layout()
        plt.savefig("longest_chain_ratio_by_cpu_speed.png")
        plt.show()
        
    def plot_branch_count_per_peer(self):
        data = []
        for peer_id, info in self.branch_lengths.items():
            data.append({
                'Peer ID': peer_id,
                'Branch Count': len(info['lengths']),
                'CPU': info['cpu'],
                'Speed': info['speed']
            })

        df = pd.DataFrame(data)
        df = df.sort_values(by="Peer ID")

        plt.figure(figsize=(12, 6))
        colors = {
            ('high CPU', 'fast'): 'blue',
            ('high CPU', 'slow'): 'orange',
            ('low CPU', 'fast'): 'green',
            ('low CPU', 'slow'): 'red'
        }
        bar_colors = [colors[(row['CPU'], row['Speed'])] for _, row in df.iterrows()]

        plt.bar(df['Peer ID'], df['Branch Count'], color=bar_colors)
        plt.xlabel("Peer ID")
        plt.ylabel("Number of Branches")
        plt.title("Number of Branches per Peer (colored by CPU and Speed)")
        legend_handles = [plt.Line2D([0], [0], color=clr, lw=6, label=f"{cpu}, {spd}") for (cpu, spd), clr in colors.items()]
        plt.legend(handles=legend_handles)
        plt.tight_layout()
        plt.savefig("bar_chart_branch_count_per_peer.png")
        plt.show()

    def plot_branch_length_distribution_by_type(self):
        records = []
        for peer_id, info in self.branch_lengths.items():
            for length in info['lengths']:
                records.append({
                    'Peer ID': peer_id,
                    'Branch Length': length,
                    'CPU': info['cpu'],
                    'Speed': info['speed'],
                    'Type': f"{info['cpu']}, {info['speed']}"
                })

        if not records:
            print("No branch lengths to plot.")
            return

        df = pd.DataFrame(records)

        plt.figure(figsize=(12, 6))
        sns.boxplot(x="Type", y="Branch Length", data=df,hue="Type", palette="Set2")
        plt.title("Branch Length Distribution by Peer Type")
        plt.xlabel("Peer Type (CPU, Speed)")
        plt.ylabel("Branch Length (in blocks)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("boxplot_branch_length_by_type.png")
        plt.show()

    def plot_block_tree(self, peer_id, format='pdf'):
        """
        Plots the block tree of a specific peer using Graphviz with high clarity.
        """
        peer = self.sim.network.peers[peer_id]
        tree = peer.block_tree

        dot = Digraph(comment=f"Peer {peer_id} Block Tree")

        # Set high resolution & large figure size for long chains
        dot.attr(
            size='30,10',           # canvas size (width,height) in inches
            rankdir='LR',           # left to right layout
            splines='true',
            nodesep='0.4',          # spacing between nodes
            ranksep='0.6',          # spacing between ranks (layers)
            dpi='1000'               # high DPI
        )
        dot.attr('node', shape='ellipse', fontsize='18')  # large font

        for blk_id, block in tree.blocks.items():
            label = blk_id if blk_id == "blk_0" else blk_id.split("_")[1]
            dot.node(blk_id, label=label)

        for blk_id, block in tree.blocks.items():
            if block.parent_id:
                dot.edge(block.parent_id, blk_id)

        output_path = f"peer_{peer_id}_block_tree"
        dot.render(output_path, format=format, cleanup=True)
        print(f"Block tree for Peer {peer_id} saved as {output_path}.{format}")

    def plot_all_block_trees(self, format='png'):
        for peer in self.sim.network.peers:
            self.plot_block_tree(peer.peer_id, format=format)
    
    def run_all(self):
        print("Analyzing Longest Chain Contributions...")
        self.analyze_longest_chain_contribution()
        self.plot_longest_chain_ratios()

        print("Analyzing Branch Lengths...")
        self.analyze_branch_lengths()
        self.plot_branch_count_per_peer()
        self.plot_branch_length_distribution_by_type()
        self.plot_all_block_trees()
        print("Analysis complete.")


# ------------------------------
# Main Function
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Discrete-event crypto network simulator")
    parser.add_argument("z0", type=float, help="Percentage of slow nodes")
    parser.add_argument("z1", type=float, help="Percentage of low CPU nodes")
    parser.add_argument("--Ttx", type=float, default=10, help="Mean inter-txn time (sec)")
    parser.add_argument("--I", type=float, default=600, help="Mean block interval (sec)")
    parser.add_argument("--n", type=int, default=10, help="Number of peers")
    parser.add_argument("--end", type=int, default=3600, help="Simulation end time (sec)")

    args = parser.parse_args()

  
    sim = Simulator(n=args.n, z0=args.z0, z1=args.z1, t_tx=args.Ttx, block_interval=args.I)
    sim.initialize()
    sim.run(end_time=args.end)

    dump_block_trees(sim)
    BlockchainAnalyzer(sim).run_all()

if __name__ == "__main__":
    main()





