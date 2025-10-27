#!/usr/bin/env python3
"""
Test Multi-Process gRPC Server Performance
Measures throughput and latency with multiple server instances
"""

import asyncio
import time
import statistics
import argparse
from typing import List, Tuple

import grpc
from grpc import aio

try:
    import optimized_lipsyncsrv_pb2
    import optimized_lipsyncsrv_pb2_grpc
except ImportError:
    print("❌ Error: gRPC stubs not found!")
    print("Generate them with:")
    print("  python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. optimized_lipsyncsrv.proto")
    import sys
    sys.exit(1)


class MultiProcessTester:
    """Test multiple gRPC server instances"""
    
    def __init__(self, ports: List[int]):
        self.ports = ports
        self.channels = []
        self.stubs = []
    
    async def connect_all(self):
        """Connect to all servers"""
        print(f"\n🔌 Connecting to {len(self.ports)} servers...")
        
        for port in self.ports:
            channel = aio.insecure_channel(
                f'localhost:{port}',
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ]
            )
            stub = optimized_lipsyncsrv_pb2_grpc.OptimizedLipSyncServiceStub(channel)
            
            # Test connection
            try:
                health = await stub.HealthCheck(optimized_lipsyncsrv_pb2.HealthRequest())
                print(f"   ✅ localhost:{port} - {health.status} ({health.loaded_models} models)")
                self.channels.append(channel)
                self.stubs.append(stub)
            except Exception as e:
                print(f"   ❌ localhost:{port} - Failed: {e}")
        
        print(f"\n✅ Connected to {len(self.stubs)}/{len(self.ports)} servers")
    
    async def test_single_server(self, server_idx: int, num_requests: int = 20):
        """Test throughput of single server"""
        stub = self.stubs[server_idx]
        port = self.ports[server_idx]
        
        print(f"\n📊 Testing server {server_idx + 1} (port {port})...")
        
        latencies = []
        successes = 0
        
        start_time = time.time()
        
        for i in range(num_requests):
            frame_id = i % 523
            req_start = time.time()
            
            request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
                model_name='sanders',
                frame_id=frame_id
            )
            
            response = await stub.GenerateInference(request)
            
            req_time = (time.time() - req_start) * 1000
            latencies.append(req_time)
            
            if response.success:
                successes += 1
        
        total_time = time.time() - start_time
        
        return {
            'port': port,
            'latencies': latencies,
            'successes': successes,
            'total_requests': num_requests,
            'total_time': total_time,
            'throughput': num_requests / total_time
        }
    
    async def test_all_concurrent(self, requests_per_server: int = 20):
        """Test all servers concurrently"""
        print(f"\n🌊 Testing ALL {len(self.stubs)} servers concurrently...")
        print(f"   {requests_per_server} requests per server = {requests_per_server * len(self.stubs)} total")
        
        start_time = time.time()
        
        # Create tasks for all servers
        tasks = [
            self.test_single_server(i, requests_per_server)
            for i in range(len(self.stubs))
        ]
        
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        return results, total_time
    
    async def test_round_robin(self, total_requests: int = 100):
        """Test round-robin distribution across servers"""
        print(f"\n🔄 Testing round-robin distribution ({total_requests} requests)...")
        
        latencies = []
        successes = 0
        
        start_time = time.time()
        
        for i in range(total_requests):
            server_idx = i % len(self.stubs)
            stub = self.stubs[server_idx]
            frame_id = i % 523
            
            req_start = time.time()
            
            request = optimized_lipsyncsrv_pb2.OptimizedInferenceRequest(
                model_name='sanders',
                frame_id=frame_id
            )
            
            response = await stub.GenerateInference(request)
            
            req_time = (time.time() - req_start) * 1000
            latencies.append(req_time)
            
            if response.success:
                successes += 1
        
        total_time = time.time() - start_time
        
        return {
            'latencies': latencies,
            'successes': successes,
            'total_requests': total_requests,
            'total_time': total_time,
            'throughput': total_requests / total_time
        }
    
    async def close_all(self):
        """Close all connections"""
        for channel in self.channels:
            await channel.close()
    
    def print_results(self, results, total_time):
        """Print test results"""
        print("\n" + "=" * 80)
        print("📊 RESULTS")
        print("=" * 80)
        
        all_latencies = []
        total_successes = 0
        total_requests = 0
        
        for i, result in enumerate(results):
            print(f"\n🖥️  Server {i + 1} (port {result['port']}):")
            print(f"   Requests: {result['successes']}/{result['total_requests']} successful")
            print(f"   Latency: {statistics.mean(result['latencies']):.1f}ms avg, "
                  f"{min(result['latencies']):.1f}ms min, {max(result['latencies']):.1f}ms max")
            print(f"   Throughput: {result['throughput']:.1f} FPS")
            
            all_latencies.extend(result['latencies'])
            total_successes += result['successes']
            total_requests += result['total_requests']
        
        print("\n" + "=" * 80)
        print("📈 AGGREGATE RESULTS")
        print("=" * 80)
        print(f"Total requests: {total_successes}/{total_requests} successful")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total throughput: {total_requests / total_time:.1f} FPS")
        print(f"Avg latency: {statistics.mean(all_latencies):.1f}ms")
        print(f"Min latency: {min(all_latencies):.1f}ms")
        print(f"Max latency: {max(all_latencies):.1f}ms")
        print(f"Latency stdev: {statistics.stdev(all_latencies):.1f}ms")


async def main():
    parser = argparse.ArgumentParser(description='Test multi-process gRPC servers')
    parser.add_argument('--ports', type=int, nargs='+', default=[50051, 50052, 50053, 50054],
                       help='Server ports to test')
    parser.add_argument('--port-range', type=str, default=None,
                       help='Port range in format "50051-50060" (alternative to --ports)')
    parser.add_argument('--requests', type=int, default=20,
                       help='Requests per server in concurrent test')
    parser.add_argument('--round-robin', type=int, default=100,
                       help='Total requests for round-robin test')
    
    args = parser.parse_args()
    
    # Parse port range if provided
    if args.port_range:
        try:
            start_port, end_port = map(int, args.port_range.split('-'))
            args.ports = list(range(start_port, end_port + 1))
            print(f"Using port range: {start_port}-{end_port} ({len(args.ports)} servers)")
        except ValueError:
            print(f"❌ Invalid port range format: {args.port_range}")
            print("   Expected format: 50051-50060")
            return
    
    print("=" * 80)
    print("🧪 MULTI-PROCESS GRPC SERVER PERFORMANCE TEST")
    print("=" * 80)
    
    tester = MultiProcessTester(args.ports)
    
    try:
        # Connect to all servers
        await tester.connect_all()
        
        if len(tester.stubs) == 0:
            print("\n❌ No servers available. Make sure they're running:")
            print("   PowerShell: .\\start_multi_grpc.ps1")
            return
        
        # Test 1: All servers concurrently
        print("\n" + "=" * 80)
        print("TEST 1: CONCURRENT LOAD (All servers simultaneously)")
        print("=" * 80)
        
        results, total_time = await tester.test_all_concurrent(args.requests)
        tester.print_results(results, total_time)
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test 2: Round-robin
        print("\n" + "=" * 80)
        print("TEST 2: ROUND-ROBIN DISTRIBUTION")
        print("=" * 80)
        
        rr_result = await tester.test_round_robin(args.round_robin)
        
        print(f"\nRequests: {rr_result['successes']}/{rr_result['total_requests']} successful")
        print(f"Total time: {rr_result['total_time']:.2f}s")
        print(f"Throughput: {rr_result['throughput']:.1f} FPS")
        print(f"Avg latency: {statistics.mean(rr_result['latencies']):.1f}ms")
        print(f"Min latency: {min(rr_result['latencies']):.1f}ms")
        print(f"Max latency: {max(rr_result['latencies']):.1f}ms")
        
        # Summary
        print("\n" + "=" * 80)
        print("🎯 PERFORMANCE SUMMARY")
        print("=" * 80)
        
        single_server_fps = results[0]['throughput']
        total_concurrent_fps = sum(r['throughput'] for r in results)
        speedup = total_concurrent_fps / single_server_fps
        
        print(f"\nSingle server baseline: {single_server_fps:.1f} FPS")
        print(f"Multi-process ({len(results)} servers): {total_concurrent_fps:.1f} FPS")
        print(f"Speedup: {speedup:.2f}x")
        
        if speedup >= 3.0:
            print("\n✅ EXCELLENT! Your RTX 6000 Ada has great multi-process support!")
            print("   Recommended: Use 4-6 processes for production")
        elif speedup >= 2.0:
            print("\n✅ GOOD! Multi-process helps on your GPU")
            print("   Recommended: Use 3-4 processes for production")
        elif speedup >= 1.5:
            print("\n⚠️  MODERATE gain from multi-process")
            print("   Consider batching for better throughput")
        else:
            print("\n❌ LIMITED gain from multi-process")
            print("   Batching would be more effective than multi-process")
        
        print("\n" + "=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await tester.close_all()


if __name__ == '__main__':
    asyncio.run(main())
