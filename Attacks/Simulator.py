import argparse
import random
import heapq
import matplotlib.pyplot as plt
from graphviz import Digraph
import seaborn as sns
import pandas as pd
from collections import deque
import itertools
import subprocess
import re
import networkx as nx
import numpy as np
from collections import defaultdict
import hashlib

MAX_LEAD_THRESHOLD = 5
# ----------------------------
# Event Types
# ----------------------------

TXN_GENERATE = "TXN_GENERATE"
TXN_RECEIVE = "TXN_RECEIVE"
SEND_BLOCK = "SEND_BLOCK"
RECEIVE_BLOCK = "RECEIVE_BLOCK"
HASH_ANNOUNCE = "HASH_ANNOUNCE"
GET_BLOCK = "GET_BLOCK"
TIMEOUT_EXPIRE = "TIMEOUT_EXPIRE"

# Counters for unique IDs for block and transaction
blk_counter = itertools.count(1)
txn_counter = itertools.count(1) 

# ------------------------------
# Peer Class 
# ------------------------------

class Peer:
    def __init__(self, n, z0):
        self.n = n
        self.z0 = z0

    class Node:
        def __init__(self, peer_id, speed, type):
            self.peer_id = peer_id
            self.speed = speed
            self.type = type 
            self.hash_fraction = 0
            self.balance = 100  #### CHANGEABLE ####
            self.pending_txns = [] 
            self.block_tree = {}  # For blockchain structure
            self.balance_map = {} # dict : peer_id -> balance (update the map on recieving block)
            self.block_hash_map = {} # hash -> block
            self.requested_hashes = set() 
            self.hash_timeouts = {} # block_hash -> (timeout_time, [backup_peers])
            self.private_block_tree = BlockchainTree() # Used by malicious miners
            self.private_tip = None
            self.is_ringmaster = False
            self.lead = 0 # (private - public) chain length

        # representation of node
        def __repr__(self):
            return f"Peer({self.peer_id}, {self.speed}, {self.type})"

    def generate_peers(self):
        num_slow = int((self.z0 / 100) * self.n)

        slow_ids = set(random.sample(range(0, self.n), num_slow))

        peers = []

        for i in range(0, self.n):
            speed = 'slow' if i in slow_ids else 'fast'
            type = 'honest' if i in slow_ids else 'malicious'
            peers.append(self.Node(i, speed, type))
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
        self.state = None
        self.timestamp = None

# ------------------------------
# Event Class
# ------------------------------

class Event:
    def __init__(self, timestamp, event_type, peer_id):
        self.timestamp = timestamp
        self.event_type = event_type
        self.peer_id = peer_id
        self.txn = None # Transaction associated to the event (For TXN_RECEIVE : receiver needs to store the transaction in pending transactions) 
        self.block = None            # For RECEIVE_BLOCK
        self.block_hash = None       # For HASH_ANNOUNCE, GET_BLOCK
        self.requesting_peer_id = None  # For GET_BLOCK
        self.sender_id = None

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
    def __init__(self, n, z0):
        self.n = n
        self.peers = Peer(n, z0).generate_peers()
        self.adjacency_list = {i: set() for i in range(0,self.n)}
        self.visited = set() # for BFS to check connectivity
        self.graph = nx.Graph()
        self.latencies = {}
        self.link_speeds = {} 

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
    
    def draw_topology(self):
        G = self.generate_topology()
        pos = nx.spring_layout(G, seed=42)

        honest_nodes = [p.peer_id for p in self.peers if p.type == 'honest']
        malicious_nodes = [p.peer_id for p in self.peers if p.type == 'malicious']

        nx.draw_networkx_nodes(G,pos,nodelist=honest_nodes,node_color='skyblue',label='Honest')
        nx.draw_networkx_nodes(G,pos,nodelist=malicious_nodes,node_color='orange',label='Malicious')
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_labels(G,pos)
        plt.title("Main Network")
        plt.axis('off')
        plt.legend()
        plt.show()

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
# Overlay Network Class
# ------------------------------

class OverlayNetwork:
    def __init__(self, malicious_peers):
        self.peers = malicious_peers
        self.n = len(malicious_peers)
        self.adjacency_list = {p.peer_id: set() for p in self.peers}
        self.visited = set() # for BFS to check connectivity
        self.graph = nx.Graph()
        self.latencies = {}
        self.link_speeds = {} 

    def create_adjacency_list(self):
        max_attempts = 10000 # max number of attempts to create adjacency list
        attempts = 0
        while not all(len(neighbours) >= 3 for neighbours in self.adjacency_list.values()):
            if attempts >= max_attempts:
                self.adjacency_list = {p.peer_id: set() for p in self.peers}
                attempts = 0
            i,j = random.sample(self.peers,2)
            if i.peer_id!=j.peer_id and len(self.adjacency_list[i.peer_id]) < 6 and len(self.adjacency_list[j.peer_id]) < 6:
                if j.peer_id not in self.adjacency_list[i.peer_id]:
                    self.adjacency_list[i.peer_id].add(j.peer_id)
                    self.adjacency_list[j.peer_id].add(i.peer_id)
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
            self.BFS(self.peers[0].peer_id)
            connected = (len(self.visited) == self.n)
            if not connected:
                self.adjacency_list = {p.peer_id: set() for p in self.peers}
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
    
    def draw_topology(self):
        G = self.generate_topology()
        pos = nx.spring_layout(G, seed=42,k=0.8)
        nx.draw_networkx_nodes(G,pos,node_color=['orange']*self.n)
        nx.draw_networkx_edges(G,pos)
        nx.draw_networkx_labels(G,pos)
        plt.title("Overlay Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def initialize_latencies_and_speeds(self): 
        for i in self.adjacency_list:
            for j in self.adjacency_list[i]:
                if (j,i) in self.latencies: 
                    continue
                rho = random.uniform(1,10)/1000 # speed of light propogation delay in seconds
                self.latencies[(i,j)] = rho
                self.latencies[(j,i)] = rho

                c = 100 * 1e6
                self.link_speeds[(i,j)] = c  # link speed between i and j in bits/second
                self.link_speeds[(j,i)] = c


    def compute_latency(self, i, j, message_size_bits):
        # Latency = ρ + |m| / c + d 
        if (i, j) not in self.latencies or (i, j) not in self.link_speeds:
            # No direct link: return infinite latency
            print(f"Warning: No direct overlay link between Peer {i} and Peer {j}")
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

    def add_block(self, block, time):
        self.blocks[block.blk_id] = block
        self.block_times[block.blk_id] = time

        if block.parent_id is not None:
            if block.parent_id not in self.blocks:
                # Parent is missing: do NOT try to update longest_chain_tip
                print(f"[DEBUG] Parent {block.parent_id} not found. Deferring longest chain update.")
                return  # Don't update longest_chain_tip yet

            self.children.setdefault(block.parent_id, []).append(block.blk_id)

        # Now it's safe to update longest chain
        if self.longest_chain_tip is None or self.chain_length(block.blk_id) > self.chain_length(self.longest_chain_tip):
            self.longest_chain_tip = block.blk_id

    def chain_length(self,blk_id):
        length = 0
        while blk_id:
            if blk_id not in self.blocks:
                print(f"[chain_length] ❗ Block {blk_id} missing from tree")
                break
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
    def __init__(self, n, z0, t_tx, block_interval,Tt,disable_eclipse):
        self.n = n
        self.z0 = z0
        self.t_tx = t_tx
        self.block_interval = block_interval
        self.timeout = Tt
        self.disable_eclipse = disable_eclipse
        self.event_queue = EventQueue()
        self.network = Network(n, z0)
        self.current_time = 0
        self.ringmaster_id = None

    def initialize(self):
        # generate genesis block, schedule initial events
        print("Initializing network...")
        self.network.isConnected() # Generate the adjacency list
        self.network.initialize_latencies_and_speeds()

        malicious_peers = [p for p in self.network.peers if p.type=='malicious']

        self.overlay = OverlayNetwork(malicious_peers)
        self.overlay.isConnected()
        self.overlay.initialize_latencies_and_speeds()

        # Each Peer maintains a balance map of all the peers
        for peer in self.network.peers:
            for p in self.network.peers:
                peer.balance_map[p.peer_id] = p.balance
            peer.block_tree = BlockchainTree()
            if peer in malicious_peers:
                peer.private_block_tree = BlockchainTree()
                peer.private_tip = "blk_0"
                peer.block_hash_map = {}
                peer.requested_hashes = set()
                peer.timeout_buffer = {} # hash -> timer info

        # Adding Genesis Block 
        genesis_block = Block("blk_0",None,[],-1) # Creator is -1 for genesis block
        genesis_block.state = {p.peer_id : p.balance for p in self.network.peers}
        for peer in self.network.peers:
            peer.block_tree.add_block(genesis_block,0)
            if peer in malicious_peers:
                peer.private_block_tree.add_block(genesis_block,0)
        
        if malicious_peers:
            ringmaster = random.choice(malicious_peers)
            ringmaster.is_ringmaster = True
            self.ringmaster_id = ringmaster.peer_id
            print(f"[*] Malicious Peer {self.ringmaster_id} selected as ringmaster.")

        # Assign hash fraction
        for peer in self.network.peers:
            if peer.type == 'honest':
                peer.hash_fraction = 1 / self.n
            elif peer.is_ringmaster:
                peer.hash_fraction = len(malicious_peers)/self.n
            else:
                peer.hash_fraction = 0

        # Schedule first transaction generation for all peers
        for peer in self.network.peers:
            next_time = np.random.exponential(self.t_tx)
            event = Event(timestamp=next_time, event_type=TXN_GENERATE, peer_id=peer.peer_id)
            self.event_queue.add_event(event)

        # schedule mining for each peer
        for peer in self.network.peers:
            if peer.hash_fraction > 0:
                self.schedule_block_mining(peer, self.current_time)

        print("Initialization Complete .")

    def compute_block_hash(block):
        header = f"{block.blk_id}{block.parent_id}{block.creator_id}"
        return hashlib.sha256(header.encode()).hexdigest()
    
    def release_private_chain(self, ringmaster,upto=None):
        chain = []
        blk_id = ringmaster.private_tip
        while blk_id and blk_id != 'blk_0':
            block = ringmaster.private_block_tree.blocks[blk_id]
            chain.append(block)
            blk_id = block.parent_id
        chain.reverse()  # from genesis to tip

        if upto is not None:
            chain = chain[:upto]

        print(f"[{self.current_time:.2f}s] Ringmaster releasing private chain of length {len(chain)}")

        for block in chain:
            block_hash = Simulator.compute_block_hash(block)

            # Add to hash maps of all malicious peers (to serve GET_BLOCK)
            for peer in self.network.peers:
                if peer.type == 'malicious':
                    peer.block_hash_map[block_hash] = block

            # Announce block hash to neighbors
            for peer in self.network.peers:
                if peer.type == 'malicious':
                    for neighbor_id in self.network.adjacency_list[peer.peer_id]:
                        latency = self.network.compute_latency(peer.peer_id, neighbor_id, message_size_bits=64*8)
                        announce_time = self.current_time + latency
                        event = Event(announce_time, HASH_ANNOUNCE, neighbor_id)
                        event.block_hash = block_hash
                        event.sender_id = peer.peer_id
                        self.event_queue.add_event(event)

        # Reset private chain
        for peer in self.network.peers:
            if peer.type == 'malicious':
                peer.lead = 0
                peer.private_tip = peer.block_tree.longest_chain_tip


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
        elif event.event_type == HASH_ANNOUNCE:
            self.handle_hash_announce(event.peer_id,event.block_hash, event.sender_id)
        elif event.event_type == GET_BLOCK:
            self.handle_get_block(event.peer_id, event.requesting_peer_id,event.block_hash)
        elif event.event_type == TIMEOUT_EXPIRE:
            self.handle_timeout_expiry(event.peer_id,event.block_hash)
    
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
        print(f"[{self.current_time:.2f}s] Scheduled mining for Peer {peer.peer_id}, hash_fraction = {peer.hash_fraction}")

    def handle_block_creation(self,peer_id):
        peer = self.network.peers[peer_id]
        if peer.type == 'honest':
            tip = peer.block_tree.longest_chain_tip
            seen_txns = peer.block_tree.get_chain_txns(tip) # transaction till now in the longest chain of the peer block tree --> should not be included in the new block

            block_txns = []
            total_size = 1024 # 1KB coinbase
            for txn in peer.pending_txns:
                if txn.txn_id not in seen_txns:
                    txn_size = 1024
                    if  total_size + txn_size > 8000000:  # 1MB = 8*10^6
                        break
                    if txn.isValid(peer.balance_map[txn.sender_id]):
                        block_txns.append(txn)
                        total_size += txn_size

            coinbase = Transaction(f"coinbase_{peer_id}_{self.current_time:.2f}",peer_id,peer_id,50)
            block_txns.insert(0,coinbase) # insert coinbase transaction at top of the new block
    
            blk_id = f"blk_{next(blk_counter)}"

            parent_block = peer.block_tree.blocks[tip]
            block_state = parent_block.state.copy()

            new_block = Block(blk_id,tip,block_txns,peer_id)
            new_block.state = block_state  # Attach snapshot to the block
            peer.block_tree.add_block(new_block,self.current_time)

            peer.pending_txns = [txn for txn in peer.pending_txns if txn not in block_txns] # remove the transactions added in the new block from the pending transactions

            print(f"[{self.current_time:.2f}s] Peer {peer_id} mined block {blk_id} with {len(block_txns)} txns")

            # Update balance
            for txn in block_txns:
                if txn.sender_id != txn.receiver_id: # for coinbase transaction txn.sender_id == txn.receiver_id . hence it should only be added not deducted
                    peer.balance_map[txn.sender_id] -= txn.amount
                peer.balance_map[txn.receiver_id] += txn.amount

            block_hash = Simulator.compute_block_hash(new_block)
            peer.block_hash_map[block_hash] = new_block

            # Broadcast to hash to neighbours
            for neighbour_id in self.network.adjacency_list[peer_id]:
                latency = self.network.compute_latency(peer_id,neighbour_id,message_size_bits=64*8)  # total_size is in bytes 64B hash value is propogated
                announce_time = self.current_time + latency
                event = Event(announce_time,HASH_ANNOUNCE,neighbour_id)
                event.block_hash = block_hash
                event.sender_id = peer_id
                self.event_queue.add_event(event)

        if peer.is_ringmaster:
            tip = peer.private_tip or peer.block_tree.longest_chain_tip
            if tip not in peer.private_block_tree.blocks:
                print(f"[{self.current_time:.2f}s] ❗ Ringmaster attempted to mine on missing tip {tip}")
                return
            seen_txns = peer.block_tree.get_chain_txns(tip)

            block_txns = []
            total_size = 1024 # 1KB coinbase
            for txn in peer.pending_txns:
                if txn.txn_id not in seen_txns:
                    txn_size = 1024
                    if  total_size + txn_size > 8000000:  # 1MB = 8*10^6
                        break
                    if txn.isValid(peer.balance_map[txn.sender_id]):
                        block_txns.append(txn)
                        total_size += txn_size

            coinbase = Transaction(f"coinbase_{peer_id}_{self.current_time:.2f}",peer_id,peer_id,50)
            block_txns.insert(0,coinbase) # insert coinbase transaction at top of the new block
    
            blk_id = f"blk_{next(blk_counter)}"
            parent_block = peer.block_tree.blocks[tip]
            block_state = parent_block.state.copy()
            new_block = Block(blk_id,tip,block_txns,peer_id)
            new_block.state = block_state
            peer.private_block_tree.add_block(new_block,self.current_time) 
            

            # Check successful addition
            if new_block.blk_id in peer.private_block_tree.blocks:
                peer.private_tip = new_block.blk_id
                peer.block_tree.add_block(new_block, self.current_time)
            else:
                print(f"[{self.current_time:.2f}s] ❗ Ringmaster failed to add block {new_block.blk_id} (parent missing: {new_block.parent_id})")
            peer.lead += 1 
        
            print(f"[{self.current_time:.2f}s] Ringmaster selfishly mined block {blk_id}, lead = {peer.lead}")
            
            L_public = peer.block_tree.chain_length(peer.block_tree.longest_chain_tip)
            L_private = peer.private_block_tree.chain_length(peer.private_tip)

            if peer.lead == 1 and L_public == L_private:
                self.release_private_chain(peer,upto=1)
            elif L_public == L_private - 1:
                self.release_private_chain(peer,upto=None)
            elif peer.lead >= MAX_LEAD_THRESHOLD:
                # Ringmaster is hoarding too many blocks without reward — dump all
                print(f"[{self.current_time:.2f}s] Ringmaster releasing private chain (lead={peer.lead}) due to max threshold")
                self.release_private_chain(peer)

        if peer.hash_fraction > 0:    
            self.schedule_block_mining(peer,self.current_time)
        
        print(f"[{self.current_time:.2f}s] Peer {peer_id} current chain tip: {peer.block_tree.longest_chain_tip} | Length = {peer.block_tree.chain_length(peer.block_tree.longest_chain_tip)}")

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
        
        parent_block = peer.block_tree.blocks.get(block.parent_id)
        if not parent_block or not parent_block.state:
            print(f"[{self.current_time:.2f}s] Peer {peer_id} cannot validate block {block.blk_id} — missing parent state.")
            return
        # Validate transactions
        temp_balance = parent_block.state.copy()
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
        
        block.state = temp_balance.copy()

        # Update balances
        peer.block_tree.add_block(block,self.current_time)
        if peer.type == 'malicious':
            peer.private_block_tree.add_block(block,self.current_time)
        if peer.block_tree.longest_chain_tip == block.blk_id:
            # Apply temp_balance updates to actual peer.balance
            peer.balance_map = temp_balance.copy()
            if peer.hash_fraction > 0:
                self.schedule_block_mining(peer, self.current_time)

        print(f"[{self.current_time:.2f}s] Peer {peer_id} accepted block {block.blk_id}")

        # Check if any orphan blocks can now be added
        blk_id = block.blk_id
        if blk_id in peer.block_tree.orphan_blocks:
            orphans_to_process = peer.block_tree.orphan_blocks.pop(blk_id, [])
            for orphan in orphans_to_process:
                self.handle_block_receive(peer_id, orphan)

        # Restart mining if new longest chain
        if peer.block_tree.longest_chain_tip ==block.blk_id and peer.hash_fraction > 0:
            self.schedule_block_mining(peer,self.current_time)

        block_hash = Simulator.compute_block_hash(block)
        peer.block_hash_map[block_hash] = block
        
        # Propagate to neighbors
        for neighbour_id in self.network.adjacency_list[peer_id]:
            latency = self.network.compute_latency(peer_id, neighbour_id, message_size_bits=64*8)
            announce_time = self.current_time + latency
            event = Event(announce_time, HASH_ANNOUNCE, neighbour_id)
            event.block_hash = block_hash
            event.sender_id = peer_id
            self.event_queue.add_event(event)

        print(f"[{self.current_time:.2f}s] Peer {peer_id} current chain tip: {peer.block_tree.longest_chain_tip} | Length = {peer.block_tree.chain_length(peer.block_tree.longest_chain_tip)}")

    def handle_hash_announce(self,peer_id,block_hash,sender_id):
        peer = self.network.peers[peer_id]

        if block_hash in peer.block_hash_map or block_hash in peer.requested_hashes:
            if block_hash in peer.hash_timeouts:
                # Save backup sender
                peer.hash_timeouts[block_hash][1].append(sender_id)
            return
        
        peer.requested_hashes.add(block_hash)

        timeout_time = self.current_time + self.timeout
        peer.hash_timeouts[block_hash] = (timeout_time,[]) # added the timeout_time and an empty list for backup senders

        latency = self.network.compute_latency(peer_id,sender_id,message_size_bits=64*8)
        request_time = self.current_time + latency
        event = Event(request_time,GET_BLOCK,sender_id)
        event.requesting_peer_id = peer_id
        event.block_hash = block_hash
        self.event_queue.add_event(event)

        # Schedule a Timeout expire event
        timeout_event = Event(timeout_time,TIMEOUT_EXPIRE,peer_id)
        timeout_event.block_hash = block_hash
        self.event_queue.add_event(timeout_event)

    def handle_get_block(self,peer_id,requesting_peer_id,block_hash):
        sender = self.network.peers[peer_id] # Node which will send the block
        receiver = self.network.peers[requesting_peer_id] # Node which requests the block
        block = sender.block_hash_map.get(block_hash)

        if not block:
            print(f"[{self.current_time:.2f}s] Peer {peer_id} does NOT have block for hash {block_hash}")
            return
        
        if sender.type == 'malicious' and not self.disable_eclipse:
            original_creator = self.network.peers[block.creator_id]
            if original_creator.type == 'honest':
                print(f"[{self.current_time:.2f}s] Malicious Peer {peer_id} withheld honest block from Peer {requesting_peer_id}")
                return
        
        latency = self.network.compute_latency(peer_id,requesting_peer_id,message_size_bits=8*len(block.transactions)*1024)
        deliver_time = self.current_time + latency
        event = Event(deliver_time,RECEIVE_BLOCK,requesting_peer_id)
        event.block = block
        self.event_queue.add_event(event)

    def handle_timeout_expiry(self,peer_id,block_hash):
        peer = self.network.peers[peer_id]

        # If block has already arrived, do nothing
        if block_hash in peer.block_hash_map:
            peer.hash_timeouts.pop(block_hash,None)
            return
        
        if block_hash in peer.hash_timeouts:
            _, backup_senders = peer.hash_timeouts[block_hash]
            if backup_senders:
                next_sender = backup_senders.pop(0)
                latency = self.network.compute_latency(peer_id, next_sender, message_size_bits=64 * 8)
                request_time = self.current_time + latency
                event = Event(request_time, GET_BLOCK, next_sender)
                event.requesting_peer_id = peer_id
                event.block_hash = block_hash
                self.event_queue.add_event(event)

                # Schedule another timeout
                new_timeout = self.current_time + self.timeout
                peer.hash_timeouts[block_hash] = (new_timeout, backup_senders)
                timeout_event = Event(new_timeout, TIMEOUT_EXPIRE, peer_id)
                timeout_event.block_hash = block_hash
                self.event_queue.add_event(timeout_event)
            else:
                print(f"[{self.current_time:.2f}s] Peer {peer_id} gave up on block hash {block_hash} (no backup peers)")




def report_ringmaster_stats(sim):
    ringmaster_id = sim.ringmaster_id
    max_len = 0
    longest_peer = None

    # Find the peer with the longest chain
    for peer in sim.network.peers:
        tip = peer.block_tree.longest_chain_tip
        length = peer.block_tree.chain_length(tip)
        if length > max_len:
            max_len = length
            longest_peer = peer

    if not longest_peer:
        print("No valid longest chain found.")
        return

    blocks = longest_peer.block_tree.blocks
    chain = longest_peer.block_tree.get_longest_chain()

    total_blocks = len(chain)
    ringmaster_blocks = [blk_id for blk_id in chain if blocks[blk_id].creator_id == ringmaster_id]

    print("\n========== Simulation Summary ==========")
    print("Ringmaster:", ringmaster_id)
    print("Hash Fraction:", sim.network.peers[ringmaster_id].hash_fraction)
    print(f"Total blocks in longest chain: {total_blocks}")
    print(f"Ringmaster blocks in longest chain: {len(ringmaster_blocks)}")
    print(f"Ringmaster share: {len(ringmaster_blocks)/total_blocks:.2%}")
    print("\nLongest Chain:")
    for blk_id in chain:
        print(f"{blk_id} <- by Peer {blocks[blk_id].creator_id}")
    return len(ringmaster_blocks)/(total_blocks-1)  # removing the genesis block

def visualize_blockchain_tree(peer, malicious_ids, filename='blockchain_tree'):
    
    tree = peer.block_tree
    blocks = tree.blocks
    dot = Digraph(comment='Blockchain Tree at Ringmaster')

    for blk_id, block in blocks.items():
        creator = block.creator_id
        color = 'red' if creator in malicious_ids else 'green'
        label = f"{blk_id}\nPeer {creator}"
        dot.node(blk_id, label=label, style='filled', fillcolor=color)

        # Connect to parent if exists
        if block.parent_id and block.parent_id in blocks:
            dot.edge(block.parent_id, blk_id)

    output_path = dot.render(filename=filename, format='png', cleanup=True)
    print(f"Blockchain tree saved to {output_path}")

def compute_metrics(sim):
    ringmaster_id = sim.ringmaster_id
    if ringmaster_id is None:
        return 0.0, 0.0

    # Longest chain
    longest_peer = max(
        sim.network.peers,
        key=lambda p: p.block_tree.chain_length(p.block_tree.longest_chain_tip)
    )
    longest_chain = longest_peer.block_tree.get_longest_chain()
    blocks = longest_peer.block_tree.blocks

    # Metric 1: Ringmaster in longest / total in longest
    total_blocks = len(longest_chain)
    ringmaster_blocks_in_longest = sum(1 for blk_id in longest_chain if blocks[blk_id].creator_id == ringmaster_id)
    metric1 = ringmaster_blocks_in_longest / total_blocks if total_blocks > 0 else 0

    # Metric 2: Ringmaster in longest / all ringmaster mined
    all_ringmaster_blocks = 0
    for peer in sim.network.peers:
        for blk in peer.block_tree.blocks.values():
            if blk.creator_id == ringmaster_id:
                all_ringmaster_blocks += 1
    metric2 = ringmaster_blocks_in_longest / all_ringmaster_blocks if all_ringmaster_blocks > 0 else 0

    return metric1, metric2


def sweep_metrics(z0_values, Tt_values, n=20, Ttx=10, I=100, end_time=1000, runs_per_setting=3):

    # Results: Dict[Tt][z0] = [metric1_avg, metric2_avg]
    results_with_eclipse = defaultdict(dict)
    results_without_eclipse = defaultdict(dict)

    for Tt in Tt_values:
        for z0 in z0_values:
            print(f"\n⏳ Running for z0 = {z0}%, Tt = {Tt} WITH Eclipse")
            m1_list, m2_list = [], []
            for _ in range(runs_per_setting):
                sim = Simulator(n=n, z0=z0, t_tx=Ttx, block_interval=I, Tt=Tt, disable_eclipse=False)
                sim.initialize()
                sim.run(end_time=end_time)
                m1, m2 = compute_metrics(sim)
                m1_list.append(m1)
                m2_list.append(m2)
            results_with_eclipse[Tt][z0] = [np.mean(m1_list), np.mean(m2_list)]

            print(f"\n⏳ Running for z0 = {z0}%, Tt = {Tt} WITHOUT Eclipse")
            m1_list, m2_list = [], []
            for _ in range(runs_per_setting):
                sim = Simulator(n=n, z0=z0, t_tx=Ttx, block_interval=I, Tt=Tt, disable_eclipse=True)
                sim.initialize()
                sim.run(end_time=end_time)
                m1, m2 = compute_metrics(sim)
                m1_list.append(m1)
                m2_list.append(m2)
            results_without_eclipse[Tt][z0] = [np.mean(m1_list), np.mean(m2_list)]

    # --- Plotting Metric 1 ---
    colors = ['red', 'orange', 'green', 'blue', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '*']

    plt.figure(figsize=(10, 6))
    for i, Tt in enumerate(Tt_values):
        z = sorted(results_with_eclipse[Tt].keys())
        y1 = [results_with_eclipse[Tt][v][0] * 100 for v in z]
        y2 = [results_without_eclipse[Tt][v][0] * 100 for v in z]

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.plot(z, y1, label=f"Tt={Tt} (With Eclipse)", linestyle='-', marker=marker, color=color)
        plt.plot(z, y2, label=f"Tt={Tt} (No Eclipse)", linestyle='--', marker=marker, color=color)

    plt.title("Metric 1: Ringmaster Blocks / Total Longest Chain")
    plt.xlabel("% Honest Nodes (z0)")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metric1_ringmaster_longest_chain.png")
    plt.show()
    # --- Plotting Metric 2 ---
    plt.figure(figsize=(10, 6))
    for i, Tt in enumerate(Tt_values):
        z = sorted(results_with_eclipse[Tt].keys())
        y1 = [results_with_eclipse[Tt][v][1] * 100 for v in z]
        y2 = [results_without_eclipse[Tt][v][1] * 100 for v in z]
    
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
    
        plt.plot(z, y1, label=f"Tt={Tt} (With Eclipse)", linestyle='-', marker=marker, color=color)
        plt.plot(z, y2, label=f"Tt={Tt} (No Eclipse)", linestyle='--', marker=marker, color=color)
    
    plt.title("Metric 2: Ringmaster Blocks / All Ringmaster Blocks Mined")
    plt.xlabel("% Honest Nodes (z0)")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metric2_ringmaster_efficiency.png")
    plt.show()

# ------------------------------
# Main Function
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Attacks Test")
    parser.add_argument("z0", type=float, nargs="?", default=None,help="Percentage of slow nodes (honest)")
    parser.add_argument("--n", type=int, default=20, help="Total number of peers")
    parser.add_argument("--Ttx", type=float, default=10, help="Mean inter-txn time (sec)")
    parser.add_argument("--I", type=float, default=600, help="Mean block interval (sec)")
    parser.add_argument("--Tt", type=float, default=10,help="Timeout for get block")
    parser.add_argument("--end", type=int, default=3600, help="Simulation end time (sec)")
    parser.add_argument('--no_eclipse', action='store_true', help='Disable eclipse attack')
    parser.add_argument('--sweep',action='store_true', help = 'Run Parameter Sweeper')

    args = parser.parse_args()

    if args.sweep:
        sweep_z0 = [0, 10, 20, 30, 40, 50, 60, 70, 80]
        timeout_values = [10,50,100]
        sweep_metrics(z0_values=sweep_z0, Tt_values=timeout_values, runs_per_setting=3)
    else:
        # Normal one-time simulation
        disable_eclipse = args.no_eclipse
        sim = Simulator(n=args.n, z0=args.z0, t_tx=args.Ttx, block_interval=args.I, Tt=args.Tt, disable_eclipse=disable_eclipse)
        sim.initialize()
        sim.run(end_time=args.end)

        report_ringmaster_stats(sim)

        malicious_ids = {peer.peer_id for peer in sim.network.peers if peer.type == 'malicious'}
        visualize_blockchain_tree(sim.network.peers[sim.ringmaster_id], malicious_ids)
if __name__ == "__main__":
    main()
