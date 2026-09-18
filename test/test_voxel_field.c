// SPDX-FileCopyrightText: 2026 Erin Catto
// SPDX-License-Identifier: MIT

#include "test_macros.h"

#include "box3d/box3d.h"
#include "box3d/collision.h"
#include "box3d/math_functions.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

// Solid below floorHeight, empty above. Border voxels follow the same rule.
static b3VoxelFieldData* MakeFloorField( int countX, int countY, int countZ, int floorHeight, bool hasBorder )
{
	int count = countX * countY * countZ;
	uint8_t* voxels = calloc( count, 1 );
	for ( int z = 0; z < countZ; ++z )
	{
		for ( int y = 0; y < countY; ++y )
		{
			for ( int x = 0; x < countX; ++x )
			{
				voxels[x + countX * ( y + countY * z )] = y < floorHeight ? 1 : 0;
			}
		}
	}

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.scale = b3Vec3_one;
	def.countX = countX;
	def.countY = countY;
	def.countZ = countZ;
	def.hasBorder = hasBorder;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );
	free( voxels );
	return field;
}

static int VoxelFieldCreate( void )
{
	b3VoxelFieldData* field = MakeFloorField( 4, 4, 4, 2, false );

	ENSURE( field->version == B3_VOXEL_FIELD_VERSION );
	ENSURE( field->hash != 0 );
	ENSURE( field->countX == 4 && field->countY == 4 && field->countZ == 4 );
	ENSURE( field->solidCount == 32 );
	ENSURE( field->hasBorder == 0 );
	ENSURE( field->materialOffset == 0 );
	ENSURE( b3GetVoxelFieldMaterialIndices( field ) == NULL );
	ENSURE( field->byteCount > (int)sizeof( b3VoxelFieldData ) );

	ENSURE_SMALL( field->aabb.lowerBound.x, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.y, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.z, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.x - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.y - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 4.0f, FLT_EPSILON );

	ENSURE( b3IsVoxelSolid( field, 0, 0, 0 ) == true );
	ENSURE( b3IsVoxelSolid( field, 3, 1, 3 ) == true );
	ENSURE( b3IsVoxelSolid( field, 0, 2, 0 ) == false );
	ENSURE( b3IsVoxelSolid( field, -1, 0, 0 ) == false );
	ENSURE( b3IsVoxelSolid( field, 0, 0, 4 ) == false );

	// Same input gives the same hash
	b3VoxelFieldData* twin = MakeFloorField( 4, 4, 4, 2, false );
	ENSURE( twin->hash == field->hash );
	ENSURE( twin->byteCount == field->byteCount );
	ENSURE( memcmp( twin, field, field->byteCount ) == 0 );

	b3DestroyVoxelField( twin );
	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldMaterials( void )
{
	uint8_t voxels[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	uint8_t materials[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

	b3VoxelFieldDef def = { 0 };
	def.voxels = voxels;
	def.materialIndices = materials;
	def.scale = (b3Vec3){ 2.0f, 1.0f, 0.5f };
	def.countX = 2;
	def.countY = 2;
	def.countZ = 2;
	b3VoxelFieldData* field = b3CreateVoxelField( &def );

	ENSURE( field->materialOffset != 0 );
	const uint8_t* stored = b3GetVoxelFieldMaterialIndices( field );
	ENSURE( stored != NULL );
	ENSURE( memcmp( stored, materials, 8 ) == 0 );
	ENSURE( field->solidCount == 8 );
	ENSURE_SMALL( field->aabb.upperBound.x - 4.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.y - 2.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 1.0f, FLT_EPSILON );

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldBorder( void )
{
	// 6x6x6 with a border: interior is 4x4x4, floor is 3 high including the border row
	b3VoxelFieldData* field = MakeFloorField( 6, 6, 6, 3, true );

	ENSURE( field->hasBorder == 1 );

	// Interior solid voxels: 4 * 4 * 2 (rows y = 1, 2 are interior; row 0 is the border)
	ENSURE( field->solidCount == 32 );

	// Bounds exclude the border
	ENSURE_SMALL( field->aabb.lowerBound.x - 1.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.lowerBound.y - 1.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.x - 5.0f, FLT_EPSILON );
	ENSURE_SMALL( field->aabb.upperBound.z - 5.0f, FLT_EPSILON );

	// Border voxels are still readable as solid so they can hide faces
	ENSURE( b3IsVoxelSolid( field, 0, 0, 0 ) == true );

	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelFieldAABB( void )
{
	b3VoxelFieldData* field = MakeFloorField( 4, 2, 4, 1, false );
	b3Transform transform = { { 10.0f, 20.0f, 30.0f }, b3Quat_identity };
	b3AABB aabb = b3ComputeVoxelFieldAABB( field, transform );
	ENSURE_SMALL( aabb.lowerBound.x - 10.0f, FLT_EPSILON );
	ENSURE_SMALL( aabb.upperBound.y - 22.0f, FLT_EPSILON );
	ENSURE_SMALL( aabb.upperBound.z - 34.0f, FLT_EPSILON );
	b3DestroyVoxelField( field );
	return 0;
}

static int VoxelWave( void )
{
	b3Vec3 scale = b3Vec3_one;
	b3VoxelFieldData* a = b3CreateVoxelWave( 10, 8, 10, 0, 0, scale, 0.3f, 0.5f, true );
	b3VoxelFieldData* b = b3CreateVoxelWave( 10, 8, 10, 8, 0, scale, 0.3f, 0.5f, true );

	// Field b starts 8 interior voxels to the +x of field a. Its left border column must equal
	// a's last interior column, and a's right border column must equal b's first interior column.
	for ( int z = 0; z < 10; ++z )
	{
		for ( int y = 0; y < 8; ++y )
		{
			ENSURE( b3IsVoxelSolid( b, 0, y, z ) == b3IsVoxelSolid( a, 8, y, z ) );
			ENSURE( b3IsVoxelSolid( a, 9, y, z ) == b3IsVoxelSolid( b, 1, y, z ) );
		}
	}

	ENSURE( a->solidCount > 0 );
	b3DestroyVoxelField( a );
	b3DestroyVoxelField( b );
	return 0;
}

int VoxelFieldTest( void )
{
	RUN_SUBTEST( VoxelFieldCreate );
	RUN_SUBTEST( VoxelFieldMaterials );
	RUN_SUBTEST( VoxelFieldBorder );
	RUN_SUBTEST( VoxelFieldAABB );
	RUN_SUBTEST( VoxelWave );

	return 0;
}
